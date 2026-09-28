#include "reasoning-penalty.h"
#include "common.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

// marker words from Lotfi et al. 2026, "Quantized Reasoning Models Think They Need to Think Longer, but They Do Not"
static const char * const common_reasoning_penalty_words_default[] = {
    "perhaps", "maybe", "wait", "Wait", "actually", "hold", "Hmm", "hmm",
    "Alternatively", "alternatively", "However", "however", "instead", "Instead", "But",
    "but", "though", "although", "yet", "rather", "unless", "otherwise", "nonetheless",
    "nevertheless", "regardless", "still", "anyway", "Or", "or", "either", "whether",
    "uncertain", "unsure", "possibly", "might", "could", "another", "different", "reconsider",
    "rethink", "backtrack", "retry", "recheck", "revisit", "doubt", "confused", "wrong",
    "mistake", "error", "incorrect",
};

std::vector<std::string> common_reasoning_penalty_default_words() {
    return std::vector<std::string>(std::begin(common_reasoning_penalty_words_default), std::end(common_reasoning_penalty_words_default));
}

std::vector<common_reasoning_penalty_entry> common_reasoning_penalty_resolve(
        const struct llama_vocab       * vocab,
        const std::vector<std::string> & words,
        std::vector<std::string>       * dropped) {
    std::vector<common_reasoning_penalty_entry> res;

    // tokenize after a newline and remove the newline tokens, so the SPM dummy space prefix does not change the text
    const auto nl = common_tokenize(vocab, "\n", false, false);

    auto add = [&](const std::string & text, bool line_start_only, bool trigger) {
        auto tokens = common_tokenize(vocab, "\n" + text, false, false);
        if (tokens.size() > nl.size() && std::equal(nl.begin(), nl.end(), tokens.begin())) {
            tokens.erase(tokens.begin(), tokens.begin() + nl.size());
        } else {
            // the newline merged with the text, e.g. the text starts with a newline
            tokens = common_tokenize(vocab, text, false, false);
        }
        if (tokens.size() != 1) {
            if (dropped) {
                dropped->push_back(text);
            }
            return;
        }
        for (const auto & e : res) {
            if (e.token == tokens[0]) {
                return;
            }
        }
        res.push_back({ tokens[0], line_start_only, trigger });
    };

    for (const auto & word : words) {
        if (word.empty()) {
            continue;
        }
        if (word[0] == '=') {
            std::string piece = word.substr(1);
            string_process_escapes(piece);
            if (!piece.empty()) {
                const auto pos = piece.find_first_not_of(" \t\r\n");
                add(piece, false, pos != std::string::npos && std::isupper((unsigned char) piece[pos]) != 0);
            }
            continue;
        }
        add(" " + word, false, std::isupper((unsigned char) word[0]) != 0);
        add(word, true, true);
    }

    return res;
}

// fixed after init, shared by clones
struct common_reasoning_penalty_data {
    const llama_vocab * vocab;

    std::vector<common_reasoning_penalty_entry> entries;
    std::vector<std::string>                    labels;

    float   start;
    float   step;
    float   max;
    int32_t window;

    llama_token end_token; // tracked: first token of the end tag
    llama_token nl_single; // tracked at sent positions: "\n" and "\n\n"
    llama_token nl_double;
};

// probabilities computed in apply(), committed to the stats in accept()
struct common_reasoning_penalty_pending {
    bool  valid    = false;
    float p_end    = -1.0f; // < 0: not computed
    float p_single = -1.0f;
    float p_double = -1.0f;
};

struct common_reasoning_penalty_ctx {
    std::shared_ptr<const common_reasoning_penalty_data> data;

    // current block
    int32_t             n;           // triggers in the block, also counted but not used when window > 0
    int64_t             pos;         // tokens accepted in the block
    std::deque<int64_t> trigger_pos; // positions of triggers in the window (window > 0)
    bool                line_start;  // previous token ends with '\n', or no token yet in the block

    common_reasoning_penalty_pos     cls;     // class of the next position
    common_reasoning_penalty_pending pending;

    // cumulative stats
    int64_t              triggers;
    int64_t              markers;
    int64_t              tokens_in_think;
    std::vector<int64_t> hits;

    common_reasoning_penalty_p_stats                 p_end[COMMON_REASONING_PENALTY_POS_COUNT];
    std::vector<common_reasoning_penalty_end_sample> p_end_trace;

    int64_t                                         nl_n;
    double                                          nl_sum_single;
    double                                          nl_sum_double;
    std::vector<common_reasoning_penalty_nl_sample> sent_trace;

    std::vector<common_reasoning_penalty_entry> to_search;

    int32_t get_n() const {
        return data->window > 0 ? (int32_t) trigger_pos.size() : n;
    }

    void new_block() {
        n          = 0;
        pos        = 0;
        line_start = true;
        cls        = COMMON_REASONING_PENALTY_POS_PARA;
        pending    = {};
        trigger_pos.clear();
    }

    void reset_stats() {
        triggers        = 0;
        markers         = 0;
        tokens_in_think = 0;
        std::fill(hits.begin(), hits.end(), 0);

        for (auto & s : p_end) {
            s = {};
        }
        p_end_trace.clear();

        nl_n          = 0;
        nl_sum_single = 0.0;
        nl_sum_double = 0.0;
        sent_trace.clear();
    }

    // commit the probabilities of the position whose token is being accepted
    void commit_pending() {
        if (!pending.valid) {
            return;
        }

        if (pending.p_end >= 0.0f) {
            auto & s = p_end[cls];
            s.n++;
            s.sum += pending.p_end;
            s.max  = std::max(s.max, pending.p_end);

            const bool is_para = cls == COMMON_REASONING_PENALTY_POS_PARA;
            if ((is_para || cls == COMMON_REASONING_PENALTY_POS_LINE) && p_end_trace.size() < COMMON_REASONING_PENALTY_TRACE_MAX) {
                p_end_trace.push_back({ tokens_in_think, is_para ? 'p' : 'l', pending.p_end });
            }
        }

        if (pending.p_single >= 0.0f) {
            nl_n++;
            nl_sum_single += pending.p_single;
            nl_sum_double += pending.p_double;

            if (sent_trace.size() < COMMON_REASONING_PENALTY_TRACE_MAX) {
                sent_trace.push_back({ tokens_in_think, pending.p_single, pending.p_double });
            }
        }

        pending = {};
    }
};

static common_reasoning_penalty_pos common_reasoning_penalty_classify(const std::string & piece, common_reasoning_penalty_pos prev) {
    const auto end = piece.find_last_not_of(" \t");
    if (end == std::string::npos) {
        return COMMON_REASONING_PENALTY_POS_MID;
    }

    const std::string s = piece.substr(0, end + 1);

    const char c = s.back();
    if (c == '\n') {
        if (s.size() >= 2 && s[s.size() - 2] == '\n') {
            return COMMON_REASONING_PENALTY_POS_PARA;
        }
        // a whitespace only "\n" token after a line end makes a paragraph break
        const bool only_ws = s.find_first_not_of(" \t\r\n") == std::string::npos;
        if (only_ws && (prev == COMMON_REASONING_PENALTY_POS_LINE || prev == COMMON_REASONING_PENALTY_POS_PARA)) {
            return COMMON_REASONING_PENALTY_POS_PARA;
        }
        return COMMON_REASONING_PENALTY_POS_LINE;
    }
    if (c == '.' || c == '?' || c == '!' || c == ':') {
        return COMMON_REASONING_PENALTY_POS_SENT;
    }
    return COMMON_REASONING_PENALTY_POS_MID;
}

// returns the logit of a token, or nullptr if it is not in the candidates
static const float * common_reasoning_penalty_find_logit(const llama_token_data_array * cur_p, llama_token token) {
    if (token >= 0 && cur_p->size > (size_t) token && cur_p->data[token].id == token) {
        return &cur_p->data[token].logit;
    }
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].id == token) {
            return &cur_p->data[i].logit;
        }
    }
    return nullptr;
}

const char * common_reasoning_penalty_pos_name(enum common_reasoning_penalty_pos pos) {
    switch (pos) {
        case COMMON_REASONING_PENALTY_POS_PARA: return "para";
        case COMMON_REASONING_PENALTY_POS_LINE: return "line";
        case COMMON_REASONING_PENALTY_POS_SENT: return "sent";
        case COMMON_REASONING_PENALTY_POS_MID:  return "mid";
        default:                                return "?";
    }
}

static const char * common_reasoning_penalty_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-penalty";
}

static void common_reasoning_penalty_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_penalty_ctx *) smpl->ctx;
    const auto & data = *ctx->data;

    // before the counters move: the trace records the tokens before this position
    ctx->commit_pending();

    ctx->tokens_in_think++;

    for (size_t i = 0; i < data.entries.size(); i++) {
        const auto & e = data.entries[i];
        if (e.token != token) {
            continue;
        }
        if (e.line_start_only && !ctx->line_start) {
            break;
        }
        ctx->markers++;
        ctx->hits[i]++;
        if (e.trigger) {
            ctx->triggers++;
            ctx->n++;
            if (data.window > 0) {
                ctx->trigger_pos.push_back(ctx->pos);
            }
        }
        break;
    }

    ctx->pos++;

    if (data.window > 0) {
        while (!ctx->trigger_pos.empty() && ctx->trigger_pos.front() < ctx->pos - data.window) {
            ctx->trigger_pos.pop_front();
        }
    }

    // without a vocab only the first token of a block counts as line start
    ctx->line_start = false;
    if (data.vocab != nullptr) {
        const std::string piece = common_token_to_piece(data.vocab, token, false);
        ctx->line_start = !piece.empty() && piece.back() == '\n';
        ctx->cls        = common_reasoning_penalty_classify(piece, ctx->cls);
    } else {
        ctx->cls = COMMON_REASONING_PENALTY_POS_MID;
    }
}

// compute the tracked probabilities from the logits as they arrive, before the penalty
static void common_reasoning_penalty_track(common_reasoning_penalty_ctx * ctx, const llama_token_data_array * cur_p) {
    const auto & data = *ctx->data;

    ctx->pending = {};

    const bool track_nl = ctx->cls == COMMON_REASONING_PENALTY_POS_SENT && data.nl_single != LLAMA_TOKEN_NULL;
    if ((data.end_token == LLAMA_TOKEN_NULL && !track_nl) || cur_p->size == 0) {
        return;
    }

    float max_logit = -INFINITY;
    for (size_t i = 0; i < cur_p->size; ++i) {
        max_logit = std::max(max_logit, cur_p->data[i].logit);
    }
    if (!std::isfinite(max_logit)) {
        return;
    }

    double sum = 0.0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        sum += std::exp((double) cur_p->data[i].logit - max_logit);
    }
    const double lse = max_logit + std::log(sum);

    auto prob = [&](llama_token token) {
        const float * logit = common_reasoning_penalty_find_logit(cur_p, token);
        return logit ? (float) std::exp((double) *logit - lse) : 0.0f;
    };

    if (data.end_token != LLAMA_TOKEN_NULL) {
        ctx->pending.p_end = prob(data.end_token);
    }
    if (track_nl) {
        ctx->pending.p_single = prob(data.nl_single);
        ctx->pending.p_double = prob(data.nl_double);
    }
    ctx->pending.valid = true;
}

static void common_reasoning_penalty_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_penalty_ctx *) smpl->ctx;
    const auto & data = *ctx->data;

    common_reasoning_penalty_track(ctx, cur_p);

    const float penalty = std::min(data.max, data.start + data.step * ctx->get_n());
    if (penalty == 0.0f) {
        return;
    }

    ctx->to_search.clear();

    for (const auto & e : data.entries) {
        if (e.line_start_only && !ctx->line_start) {
            continue;
        }
        if (e.token >= 0 && cur_p->size > (size_t) e.token && cur_p->data[e.token].id == e.token) {
            cur_p->data[e.token].logit -= penalty;
        } else {
            ctx->to_search.push_back(e);
        }
    }

    if (ctx->to_search.empty()) {
        return;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        for (const auto & e : ctx->to_search) {
            if (cur_p->data[i].id == e.token) {
                cur_p->data[i].logit -= penalty;
                break;
            }
        }
    }
}

static void common_reasoning_penalty_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_penalty_ctx *) smpl->ctx;

    ctx->new_block();
    ctx->reset_stats();
}

static struct llama_sampler * common_reasoning_penalty_clone(const struct llama_sampler * smpl);

static void common_reasoning_penalty_free(struct llama_sampler * smpl) {
    delete (common_reasoning_penalty_ctx *) smpl->ctx;
}

// the shared data is not copied, only the block state and the stats
static void common_reasoning_penalty_copy_state(const struct llama_sampler * src, struct llama_sampler * dst) {
    *(common_reasoning_penalty_ctx *) dst->ctx = *(const common_reasoning_penalty_ctx *) src->ctx;
}

static struct llama_sampler_i common_reasoning_penalty_i = {
    /* .name              = */ common_reasoning_penalty_name,
    /* .accept            = */ common_reasoning_penalty_accept,
    /* .apply             = */ common_reasoning_penalty_apply,
    /* .reset             = */ common_reasoning_penalty_reset,
    /* .clone             = */ common_reasoning_penalty_clone,
    /* .free              = */ common_reasoning_penalty_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ common_reasoning_penalty_copy_state,
};

static struct llama_sampler * common_reasoning_penalty_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_penalty_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_penalty_i,
        /* .ctx   = */ new common_reasoning_penalty_ctx(*ctx)
    );
}

struct llama_sampler * common_reasoning_penalty_init(
        const struct llama_vocab                  * vocab,
        std::vector<common_reasoning_penalty_entry> entries,
        float                                       start,
        float                                       step,
        float                                       max,
        int32_t                                     window,
        llama_token                                 end_token) {
    std::vector<std::string> labels;
    labels.reserve(entries.size());
    for (const auto & e : entries) {
        std::string label = vocab ? common_token_to_piece(vocab, e.token, true) : std::to_string(e.token);
        if (e.line_start_only) {
            label += "/line";
        }
        labels.push_back(std::move(label));
    }

    // "\n" and "\n\n" must both be single tokens, else the sent tracking is off
    llama_token nl_single = LLAMA_TOKEN_NULL;
    llama_token nl_double = LLAMA_TOKEN_NULL;
    if (vocab != nullptr) {
        const auto t1 = common_tokenize(vocab, "\n",   false, false);
        const auto t2 = common_tokenize(vocab, "\n\n", false, false);
        if (t1.size() == 1 && t2.size() == 1 && t1[0] != t2[0]) {
            nl_single = t1[0];
            nl_double = t2[0];
        }
    }

    const size_t n_entries = entries.size();

    auto data = std::make_shared<const common_reasoning_penalty_data>(common_reasoning_penalty_data {
        /* .vocab     = */ vocab,
        /* .entries   = */ std::move(entries),
        /* .labels    = */ std::move(labels),
        /* .start     = */ start,
        /* .step      = */ step,
        /* .max       = */ max,
        /* .window    = */ std::max(0, window),
        /* .end_token = */ end_token,
        /* .nl_single = */ nl_single,
        /* .nl_double = */ nl_double,
    });

    auto * ctx = new common_reasoning_penalty_ctx {};
    ctx->data = std::move(data);
    ctx->hits.resize(n_entries, 0);
    ctx->new_block();
    ctx->reset_stats();

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_penalty_i,
        /* .ctx   = */ ctx
    );
}

void common_reasoning_penalty_new_block(struct llama_sampler * smpl) {
    if (!smpl) {
        return;
    }

    ((common_reasoning_penalty_ctx *) smpl->ctx)->new_block();
}

int32_t common_reasoning_penalty_get_n(const struct llama_sampler * smpl) {
    if (!smpl) {
        return 0;
    }

    return ((const common_reasoning_penalty_ctx *) smpl->ctx)->get_n();
}

common_reasoning_penalty_stats common_reasoning_penalty_get_stats(const struct llama_sampler * smpl) {
    common_reasoning_penalty_stats res;
    if (!smpl) {
        return res;
    }

    const auto * ctx = (const common_reasoning_penalty_ctx *) smpl->ctx;

    res.triggers        = ctx->triggers;
    res.markers         = ctx->markers;
    res.tokens_in_think = ctx->tokens_in_think;

    for (size_t i = 0; i < ctx->hits.size(); i++) {
        if (ctx->hits[i] > 0) {
            res.hits.emplace_back(ctx->data->labels[i], ctx->hits[i]);
        }
    }

    for (int i = 0; i < COMMON_REASONING_PENALTY_POS_COUNT; i++) {
        res.p_end[i] = ctx->p_end[i];
    }
    res.p_end_trace = ctx->p_end_trace;

    res.nl_n          = ctx->nl_n;
    res.nl_sum_single = ctx->nl_sum_single;
    res.nl_sum_double = ctx->nl_sum_double;
    res.sent_trace    = ctx->sent_trace;

    return res;
}
