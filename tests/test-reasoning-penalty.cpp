#include "reasoning-penalty.h"
#include "sampling.h"
#include "common.h"

#include "llama.h"
#include "ggml.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static const size_t N_VOCAB = 32;

// apply the sampler to an array of zero logits and return the per-token change
static std::vector<float> get_deltas(llama_sampler * smpl, bool shuffle = false) {
    std::vector<llama_token_data> cur;
    for (size_t i = 0; i < N_VOCAB; i++) {
        cur.push_back({ (llama_token) i, 0.0f, 0.0f });
    }
    if (shuffle) {
        std::mt19937 rng(42);
        std::shuffle(cur.begin(), cur.end(), rng);
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };

    llama_sampler_apply(smpl, &cur_p);

    std::vector<float> res(N_VOCAB, 0.0f);
    for (const auto & td : cur) {
        res[td.id] = td.logit;
    }
    return res;
}

static bool feq(float a, float b) {
    return std::fabs(a - b) < 1e-6f;
}

// markers: 5 = trigger, 7 = non-trigger, 9 = line start only (trigger)
static std::vector<common_reasoning_penalty_entry> make_entries() {
    return {
        { 5, false, true  },
        { 7, false, false },
        { 9, true,  true  },
    };
}

static void test_apply() {
    auto * smpl = common_reasoning_penalty_init(nullptr, make_entries(), 1.0f, 0.5f, 2.0f, 0);

    for (bool shuffle : { false, true }) {
        const auto d = get_deltas(smpl, shuffle);
        for (size_t i = 0; i < N_VOCAB; i++) {
            const float expected = (i == 5 || i == 7 || i == 9) ? -1.0f : 0.0f;
            GGML_ASSERT(feq(d[i], expected));
        }
    }

    llama_sampler_free(smpl);
    fprintf(stderr, "  Test 'apply' passed\n");
}

static void test_escalation() {
    auto * smpl = common_reasoning_penalty_init(nullptr, make_entries(), 1.0f, 0.5f, 2.0f, 0);

    llama_sampler_accept(smpl, 5); // trigger
    GGML_ASSERT(common_reasoning_penalty_get_n(smpl) == 1);
    GGML_ASSERT(feq(get_deltas(smpl)[5], -1.5f));

    llama_sampler_accept(smpl, 7); // marker, not a trigger
    llama_sampler_accept(smpl, 3); // not a marker
    llama_sampler_accept(smpl, 9); // bare marker, not at line start (no vocab): not counted
    GGML_ASSERT(common_reasoning_penalty_get_n(smpl) == 1);

    const auto d = get_deltas(smpl);
    GGML_ASSERT(feq(d[5], -1.5f) && feq(d[7], -1.5f) && feq(d[9], 0.0f) && feq(d[3], 0.0f));

    llama_sampler_accept(smpl, 5);
    GGML_ASSERT(feq(get_deltas(smpl)[7], -2.0f));
    llama_sampler_accept(smpl, 5);
    GGML_ASSERT(common_reasoning_penalty_get_n(smpl) == 3);
    GGML_ASSERT(feq(get_deltas(smpl)[7], -2.0f)); // capped

    auto stats = common_reasoning_penalty_get_stats(smpl);
    GGML_ASSERT(stats.triggers == 3 && stats.markers == 4 && stats.tokens_in_think == 6);

    common_reasoning_penalty_new_block(smpl);
    GGML_ASSERT(common_reasoning_penalty_get_n(smpl) == 0);
    GGML_ASSERT(feq(get_deltas(smpl)[9], -1.0f)); // line start again

    stats = common_reasoning_penalty_get_stats(smpl);
    GGML_ASSERT(stats.triggers == 3 && stats.markers == 4 && stats.tokens_in_think == 6);
    GGML_ASSERT(stats.hits.size() == 2 && stats.hits[0].first == "5" && stats.hits[0].second == 3);

    llama_sampler_free(smpl);
    fprintf(stderr, "  Test 'escalation' passed\n");
}

static void test_window() {
    auto * smpl = common_reasoning_penalty_init(nullptr, make_entries(), 0.0f, 1.0f, 10.0f, 3);

    llama_sampler_accept(smpl, 5);
    llama_sampler_accept(smpl, 1);
    llama_sampler_accept(smpl, 1);
    GGML_ASSERT(common_reasoning_penalty_get_n(smpl) == 1); // trigger is one of the last 3 tokens
    llama_sampler_accept(smpl, 1);
    GGML_ASSERT(common_reasoning_penalty_get_n(smpl) == 0);
    GGML_ASSERT(feq(get_deltas(smpl)[5], 0.0f));

    llama_sampler_free(smpl);
    fprintf(stderr, "  Test 'window' passed\n");
}

static void test_fixed() {
    auto * smpl = common_reasoning_penalty_init(nullptr, make_entries(), 2.0f, 0.0f, 3.0f, 0);

    for (int i = 0; i < 5; i++) {
        GGML_ASSERT(common_reasoning_penalty_get_n(smpl) == i);
        GGML_ASSERT(feq(get_deltas(smpl)[5], -2.0f));
        llama_sampler_accept(smpl, 5);
    }

    llama_sampler_free(smpl);
    fprintf(stderr, "  Test 'fixed' passed\n");
}

static void test_clone_copy(const llama_vocab * vocab) {
    const llama_token nl = common_tokenize(vocab, "\n", false, false)[0];

    auto * smpl = common_reasoning_penalty_init(vocab, make_entries(), 0.0f, 1.0f, 10.0f, 4);
    llama_sampler_accept(smpl, 5);
    llama_sampler_accept(smpl, 7);
    llama_sampler_accept(smpl, nl); // line start

    auto * clone = llama_sampler_clone(smpl);
    auto * copy  = common_reasoning_penalty_init(vocab, make_entries(), 0.0f, 1.0f, 10.0f, 4);
    llama_sampler_copy(smpl, copy);

    for (auto * s : { clone, copy }) {
        GGML_ASSERT(common_reasoning_penalty_get_n(s) == 1);
        GGML_ASSERT(feq(get_deltas(s)[9], -1.0f)); // line start state kept

        const auto st = common_reasoning_penalty_get_stats(s);
        GGML_ASSERT(st.triggers == 1 && st.markers == 2 && st.tokens_in_think == 3);

        // the window ring is kept: the trigger at position 0 drops out after 2 more tokens
        llama_sampler_accept(s, 1);
        GGML_ASSERT(common_reasoning_penalty_get_n(s) == 1);
        llama_sampler_accept(s, 1);
        GGML_ASSERT(common_reasoning_penalty_get_n(s) == 0);
    }

    // the original is not affected
    GGML_ASSERT(common_reasoning_penalty_get_n(smpl) == 1);
    GGML_ASSERT(common_reasoning_penalty_get_stats(smpl).tokens_in_think == 3);

    llama_sampler_free(smpl);
    llama_sampler_free(clone);
    llama_sampler_free(copy);
    fprintf(stderr, "  Test 'clone/copy' passed\n");
}

static void test_resolve(const llama_vocab * vocab) {
    std::vector<std::string> dropped;
    const auto entries = common_reasoning_penalty_resolve(vocab, { "Wait", "wait", "But", "recheck" }, &dropped);

    GGML_ASSERT(std::find(dropped.begin(), dropped.end(), " recheck") != dropped.end());

    bool has_space = false;
    bool has_bare  = false;
    for (const auto & e : entries) {
        const std::string piece = common_token_to_piece(vocab, e.token, true);
        GGML_ASSERT(common_tokenize(vocab, piece, false, false) == std::vector<llama_token>{ e.token });
        if (e.line_start_only) {
            GGML_ASSERT(piece[0] != ' ' && e.trigger);
        } else {
            GGML_ASSERT(piece[0] == ' ');
            GGML_ASSERT(e.trigger == (std::isupper((unsigned char) piece[1]) != 0));
        }
        has_space |= piece == " Wait" && !e.line_start_only;
        has_bare  |= piece == "Wait"  &&  e.line_start_only;
    }
    GGML_ASSERT(has_space && has_bare);

    // literal: not expanded, not gated, deduplicated
    const auto lit = common_reasoning_penalty_resolve(vocab, { "= Wait", "=\\n", "= Wait" }, nullptr);
    GGML_ASSERT(lit.size() == 2);
    GGML_ASSERT(common_token_to_piece(vocab, lit[0].token, true) == " Wait" && !lit[0].line_start_only && lit[0].trigger);
    GGML_ASSERT(common_token_to_piece(vocab, lit[1].token, true) == "\n" && !lit[1].trigger);

    GGML_ASSERT(common_reasoning_penalty_default_words().size() == 50);

    // on this BPE vocab the result is the same as a plain tokenization of each spelling
    {
        const auto words = common_reasoning_penalty_default_words();
        std::vector<llama_token> expected;
        for (const auto & w : words) {
            for (const auto & text : { " " + w, w }) {
                const auto tokens = common_tokenize(vocab, text, false, false);
                if (tokens.size() == 1 && std::find(expected.begin(), expected.end(), tokens[0]) == expected.end()) {
                    expected.push_back(tokens[0]);
                }
            }
        }
        std::vector<llama_token> got;
        for (const auto & e : common_reasoning_penalty_resolve(vocab, words, nullptr)) {
            got.push_back(e.token);
        }
        GGML_ASSERT(got == expected);
    }

    fprintf(stderr, "  Test 'resolve' passed\n");
}

// SPM vocabs add a dummy space prefix: the bare word must not resolve to the space variant
static void test_resolve_spm(const llama_vocab * vocab) {
    std::vector<std::string> dropped;
    const auto entries = common_reasoning_penalty_resolve(vocab, { "Wait" }, &dropped);

    GGML_ASSERT(dropped.empty() && entries.size() == 2);
    GGML_ASSERT(!entries[0].line_start_only && common_token_to_piece(vocab, entries[0].token, true) == " Wait");
    GGML_ASSERT( entries[1].line_start_only && common_token_to_piece(vocab, entries[1].token, true) == "Wait");

    fprintf(stderr, "  Test 'resolve spm' passed\n");
}

static void test_line_start(const llama_vocab * vocab) {
    const llama_token nl    = common_tokenize(vocab, "\n", false, false)[0];
    const llama_token word  = common_tokenize(vocab, " the", false, false)[0];
    const llama_token wait  = common_tokenize(vocab, "Wait", false, false)[0];
    const llama_token swait = common_tokenize(vocab, " Wait", false, false)[0];

    auto * smpl = common_reasoning_penalty_init(vocab, common_reasoning_penalty_resolve(vocab, { "Wait" }, nullptr), 1.0f, 0.0f, 1.0f, 0);

    // this test needs a vocab large enough for the full id range
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    auto deltas = [&]() {
        std::vector<llama_token_data> cur;
        for (llama_token i = 0; i < n_vocab; i++) {
            cur.push_back({ i, 0.0f, 0.0f });
        }
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(smpl, &cur_p);
        return std::make_pair(cur[wait].logit, cur[swait].logit);
    };

    GGML_ASSERT(feq(deltas().first, -1.0f)); // first token of a block
    llama_sampler_accept(smpl, word);
    GGML_ASSERT(feq(deltas().first,  0.0f));
    GGML_ASSERT(feq(deltas().second, -1.0f)); // space variant always applies
    llama_sampler_accept(smpl, wait);         // not at line start: not counted
    GGML_ASSERT(common_reasoning_penalty_get_stats(smpl).markers == 0);
    llama_sampler_accept(smpl, nl);
    GGML_ASSERT(feq(deltas().first, -1.0f));
    llama_sampler_accept(smpl, wait);         // at line start: counted
    GGML_ASSERT(common_reasoning_penalty_get_n(smpl) == 1);

    const auto stats = common_reasoning_penalty_get_stats(smpl);
    GGML_ASSERT(stats.markers == 1 && stats.hits.size() == 1 && stats.hits[0].first == "Wait/line");

    llama_sampler_free(smpl);
    fprintf(stderr, "  Test 'line start' passed\n");
}

static void test_tracking(const llama_vocab * vocab) {
    auto tok = [&](const char * text) {
        const auto tokens = common_tokenize(vocab, text, false, false);
        GGML_ASSERT(tokens.size() == 1);
        return tokens[0];
    };

    const llama_token end   = tok(" banana");
    const llama_token nl    = tok("\n");
    const llama_token nl2   = tok("\n\n");
    const llama_token dot   = tok(".");
    const llama_token x     = tok("x");
    const llama_token word  = tok(" the");
    const llama_token wait  = tok("Wait");
    const llama_token swait = tok(" Wait");

    // penalty 0/0/0: tracking only
    auto * smpl = common_reasoning_penalty_init(vocab, common_reasoning_penalty_resolve(vocab, { "Wait" }, nullptr), 0.0f, 0.0f, 0.0f, 0, end);

    // large offset to check the logsumexp, one masked token
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float   base    = 1000.0f;
    std::vector<llama_token_data> cur;
    auto apply = [&]() {
        cur.clear();
        for (llama_token i = 0; i < n_vocab; i++) {
            cur.push_back({ i, base, 0.0f });
        }
        cur[end].logit  = base + 2.0f;
        cur[nl].logit   = base + 1.0f;
        cur[nl2].logit  = base + 3.0f;
        cur[x].logit    = -INFINITY;
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(smpl, &cur_p);
    };

    // expected probabilities
    const double z        = (n_vocab - 4) + std::exp(2.0) + std::exp(1.0) + std::exp(3.0);
    const float  p_end    = (float) (std::exp(2.0) / z);
    const float  p_single = (float) (std::exp(1.0) / z);
    const float  p_double = (float) (std::exp(3.0) / z);
    auto near = [](float a, float b) { return std::fabs(a - b) <= 1e-4f * b; };

    // first token of the block: para, logits untouched
    apply();
    GGML_ASSERT(cur[end].logit == base + 2.0f && cur[wait].logit == base && cur[swait].logit == base);
    GGML_ASSERT(common_reasoning_penalty_get_stats(smpl).p_end[COMMON_REASONING_PENALTY_POS_PARA].n == 0); // not committed before accept
    llama_sampler_accept(smpl, word);
    auto st = common_reasoning_penalty_get_stats(smpl);
    GGML_ASSERT(st.p_end[COMMON_REASONING_PENALTY_POS_PARA].n == 1);
    GGML_ASSERT(near((float) st.p_end[COMMON_REASONING_PENALTY_POS_PARA].sum, p_end));
    GGML_ASSERT(near(st.p_end[COMMON_REASONING_PENALTY_POS_PARA].max, p_end));
    GGML_ASSERT(st.p_end_trace.size() == 1 && st.p_end_trace[0].tokens_in_think == 0 && st.p_end_trace[0].cls == 'p');

    // after " the": mid
    apply();
    llama_sampler_accept(smpl, dot);
    st = common_reasoning_penalty_get_stats(smpl);
    GGML_ASSERT(st.p_end[COMMON_REASONING_PENALTY_POS_MID].n == 1 && st.p_end_trace.size() == 1);

    // after ".": sent, "\n" / "\n\n" probabilities
    apply();
    llama_sampler_accept(smpl, nl);
    st = common_reasoning_penalty_get_stats(smpl);
    GGML_ASSERT(st.p_end[COMMON_REASONING_PENALTY_POS_SENT].n == 1);
    GGML_ASSERT(st.nl_n == 1 && near((float) st.nl_sum_single, p_single) && near((float) st.nl_sum_double, p_double));
    GGML_ASSERT(st.sent_trace.size() == 1 && st.sent_trace[0].tokens_in_think == 2);

    // after "\n": line, marker at line start counted, no logit change
    apply();
    GGML_ASSERT(cur[wait].logit == base);
    llama_sampler_accept(smpl, wait);
    st = common_reasoning_penalty_get_stats(smpl);
    GGML_ASSERT(st.p_end[COMMON_REASONING_PENALTY_POS_LINE].n == 1 && st.markers == 1 && st.triggers == 1);
    GGML_ASSERT(st.p_end_trace.size() == 2 && st.p_end_trace[1].cls == 'l' && st.p_end_trace[1].tokens_in_think == 3);

    // apply without accept (rejected draft) does not count, a later accept without apply does not count either
    llama_sampler * saved = llama_sampler_clone(smpl);
    apply();
    llama_sampler_copy(saved, smpl);
    llama_sampler_accept(smpl, nl2);
    st = common_reasoning_penalty_get_stats(smpl);
    GGML_ASSERT(st.p_end[COMMON_REASONING_PENALTY_POS_MID].n == 1 && st.p_end[COMMON_REASONING_PENALTY_POS_LINE].n == 1 && st.tokens_in_think == 5);

    apply();
    common_reasoning_penalty_new_block(smpl); // drops the pending position
    llama_sampler_accept(smpl, x);
    st = common_reasoning_penalty_get_stats(smpl);
    GGML_ASSERT(st.p_end[COMMON_REASONING_PENALTY_POS_PARA].n == 1 && st.tokens_in_think == 6);

    // after "x": mid, then "\n" then "\n" makes a paragraph break
    llama_sampler_accept(smpl, nl);
    llama_sampler_accept(smpl, nl);
    apply();
    llama_sampler_accept(smpl, word);
    st = common_reasoning_penalty_get_stats(smpl);
    GGML_ASSERT(st.p_end[COMMON_REASONING_PENALTY_POS_PARA].n == 2 && st.p_end_trace.back().cls == 'p');

    // after "\n\n": para
    llama_sampler_accept(smpl, nl2);
    apply();
    llama_sampler_accept(smpl, word);
    st = common_reasoning_penalty_get_stats(smpl);
    GGML_ASSERT(st.p_end[COMMON_REASONING_PENALTY_POS_PARA].n == 3);
    GGML_ASSERT(st.p_end[COMMON_REASONING_PENALTY_POS_LINE].n == 1 && st.p_end[COMMON_REASONING_PENALTY_POS_SENT].n == 1);

    llama_sampler_free(saved);
    llama_sampler_free(smpl);
    fprintf(stderr, "  Test 'tracking' passed\n");
}

static void test_gating(const llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const llama_token tag_start = common_tokenize(vocab, " apple", false, false)[0];
    const llama_token tag_end   = common_tokenize(vocab, " banana", false, false)[0];
    const llama_token swait     = common_tokenize(vocab, " Wait", false, false)[0];

    common_params_sampling params;

    // no words: no sampler
    {
        common_sampler_ptr gsmpl(common_sampler_init(model, params));
        GGML_ASSERT(common_sampler_get_reasoning_penalty(gsmpl.get()) == nullptr);
    }

    // penalty 0/0/0: the sampler exists, for tracking only
    {
        common_params_sampling p0;
        p0.reasoning_budget_start  = { tag_start };
        p0.reasoning_budget_end    = { { tag_end } };
        p0.reasoning_penalty.words = common_reasoning_penalty_default_words();
        p0.reasoning_penalty.start = 0.0f;
        p0.reasoning_penalty.step  = 0.0f;
        p0.reasoning_penalty.max   = 0.0f;

        common_sampler_ptr gsmpl(common_sampler_init(model, p0));
        GGML_ASSERT(common_sampler_get_reasoning_penalty(gsmpl.get()) != nullptr);
    }

    params.reasoning_budget_start = { tag_start };
    params.reasoning_budget_end   = { { tag_end } };
    params.reasoning_penalty.words = { "Wait" };

    common_sampler_ptr gsmpl(common_sampler_init(model, params));
    const auto * rp = common_sampler_get_reasoning_penalty(gsmpl.get());
    GGML_ASSERT(rp != nullptr);

    GGML_ASSERT(!common_sampler_reasoning_penalty_active(gsmpl.get()));
    common_sampler_accept(gsmpl.get(), swait, true); // outside: not counted
    GGML_ASSERT(common_reasoning_penalty_get_stats(rp).tokens_in_think == 0);

    common_sampler_accept(gsmpl.get(), tag_start, true);
    GGML_ASSERT(common_sampler_reasoning_penalty_active(gsmpl.get()));
    common_sampler_accept(gsmpl.get(), swait, true);
    GGML_ASSERT(common_reasoning_penalty_get_n(rp) == 1);

    // a cloned or copied common_sampler keeps the state, and counts its own hits after that
    {
        common_sampler_ptr clone(common_sampler_clone(gsmpl.get()));
        common_sampler_ptr copy(common_sampler_init(model, params));
        common_sampler_copy(gsmpl.get(), copy.get());

        for (auto * s : { clone.get(), copy.get() }) {
            const auto * srp = common_sampler_get_reasoning_penalty(s);
            GGML_ASSERT(srp != rp);
            GGML_ASSERT(common_sampler_reasoning_penalty_active(s));
            GGML_ASSERT(common_reasoning_penalty_get_n(srp) == 1);

            common_sampler_accept(s, swait, true);
            GGML_ASSERT(common_reasoning_penalty_get_n(srp) == 2);

            const auto st = common_reasoning_penalty_get_stats(srp);
            GGML_ASSERT(st.triggers == 2 && st.hits.size() == 1 && st.hits[0].second == 2);
        }

        const auto st = common_reasoning_penalty_get_stats(rp);
        GGML_ASSERT(common_reasoning_penalty_get_n(rp) == 1 && st.triggers == 1 && st.hits[0].second == 1);
    }

    common_sampler_accept(gsmpl.get(), tag_end, true);
    GGML_ASSERT(!common_sampler_reasoning_penalty_active(gsmpl.get()));
    common_sampler_accept(gsmpl.get(), swait, true);

    common_sampler_accept(gsmpl.get(), tag_start, true); // re-arm
    GGML_ASSERT(common_sampler_reasoning_penalty_active(gsmpl.get()));
    GGML_ASSERT(common_reasoning_penalty_get_n(rp) == 0);

    const auto stats = common_reasoning_penalty_get_stats(rp);
    GGML_ASSERT(stats.triggers == 1 && stats.tokens_in_think == 2);

    fprintf(stderr, "  Test 'gating' passed\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <vocab-file> [spm-vocab-file]\n", argv[0]);
        return 1;
    }

    printf("Testing reasoning penalty sampler...\n");

    test_apply();
    test_escalation();
    test_window();
    test_fixed();

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;

    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load vocab '%s'\n", argv[1]);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    test_resolve(vocab);
    test_line_start(vocab);
    test_clone_copy(vocab);
    test_tracking(vocab);
    test_gating(model);

    llama_model_free(model);

    if (argc >= 3) {
        llama_model * model_spm = llama_model_load_from_file(argv[2], mparams);
        if (model_spm == nullptr) {
            fprintf(stderr, "failed to load vocab '%s'\n", argv[2]);
            return 1;
        }

        test_resolve_spm(llama_model_get_vocab(model_spm));

        llama_model_free(model_spm);
    }

    llama_backend_free();

    printf("OK\n");

    return 0;
}
