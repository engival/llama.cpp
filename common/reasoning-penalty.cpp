#include "reasoning-penalty.h"
#include "common.h"

#include <algorithm>
#include <cctype>
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
};

struct common_reasoning_penalty_ctx {
    std::shared_ptr<const common_reasoning_penalty_data> data;

    // current block
    int32_t             n;           // triggers in the block, also counted but not used when window > 0
    int64_t             pos;         // tokens accepted in the block
    std::deque<int64_t> trigger_pos; // positions of triggers in the window (window > 0)
    bool                line_start;  // previous token ends with '\n', or no token yet in the block

    // cumulative stats
    int64_t              triggers;
    int64_t              markers;
    int64_t              tokens_in_think;
    std::vector<int64_t> hits;

    std::vector<common_reasoning_penalty_entry> to_search;

    int32_t get_n() const {
        return data->window > 0 ? (int32_t) trigger_pos.size() : n;
    }

    void new_block() {
        n          = 0;
        pos        = 0;
        line_start = true;
        trigger_pos.clear();
    }
};

static const char * common_reasoning_penalty_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-penalty";
}

static void common_reasoning_penalty_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_penalty_ctx *) smpl->ctx;
    const auto & data = *ctx->data;

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
    }
}

static void common_reasoning_penalty_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_penalty_ctx *) smpl->ctx;
    const auto & data = *ctx->data;

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

    ctx->triggers        = 0;
    ctx->markers         = 0;
    ctx->tokens_in_think = 0;
    std::fill(ctx->hits.begin(), ctx->hits.end(), 0);
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
        int32_t                                     window) {
    std::vector<std::string> labels;
    labels.reserve(entries.size());
    for (const auto & e : entries) {
        std::string label = vocab ? common_token_to_piece(vocab, e.token, true) : std::to_string(e.token);
        if (e.line_start_only) {
            label += "/line";
        }
        labels.push_back(std::move(label));
    }

    const size_t n_entries = entries.size();

    auto data = std::make_shared<const common_reasoning_penalty_data>(common_reasoning_penalty_data {
        /* .vocab   = */ vocab,
        /* .entries = */ std::move(entries),
        /* .labels  = */ std::move(labels),
        /* .start   = */ start,
        /* .step    = */ step,
        /* .max     = */ max,
        /* .window  = */ std::max(0, window),
    });

    auto * ctx = new common_reasoning_penalty_ctx {
        /* .data            = */ std::move(data),
        /* .n               = */ 0,
        /* .pos             = */ 0,
        /* .trigger_pos     = */ {},
        /* .line_start      = */ true,
        /* .triggers        = */ 0,
        /* .markers         = */ 0,
        /* .tokens_in_think = */ 0,
        /* .hits            = */ std::vector<int64_t>(n_entries, 0),
        /* .to_search       = */ {},
    };

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

    return res;
}
