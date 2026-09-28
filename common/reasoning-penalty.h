#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Reasoning penalty sampler: subtracts a penalty from the logits of marker tokens (e.g. " Wait", " But").
// It has no knowledge of the thinking state. The caller applies it and feeds it tokens only while inside a
// reasoning block (see common_sampler, which gates it on the reasoning budget sampler state).
//
// penalty = min(max, start + step * n), n = number of trigger hits in the current block
// (or in the last `window` accepted tokens if window > 0).
//
// Clones share the resolved entries and the parameters, and copy only the block state and the stats.
//
// Based on: "Quantized Reasoning Models Think They Need to Think Longer, but They Do Not" (Lotfi et al., 2026)

struct common_reasoning_penalty_entry {
    llama_token token;
    bool        line_start_only; // only penalize when the previous token ends with '\n' (or at block start)
    bool        trigger;         // a hit on this entry advances n
};

// position class, from the previous accepted token in the block
enum common_reasoning_penalty_pos {
    COMMON_REASONING_PENALTY_POS_PARA, // previous token ends with "\n\n" (or a "\n" after a line end), or first token of the block
    COMMON_REASONING_PENALTY_POS_LINE, // previous token ends with a single "\n"
    COMMON_REASONING_PENALTY_POS_SENT, // previous token ends with '.', '?', '!' or ':' (trailing spaces ignored)
    COMMON_REASONING_PENALTY_POS_MID,  // anything else
    COMMON_REASONING_PENALTY_POS_COUNT,
};

struct common_reasoning_penalty_p_stats {
    int64_t n   = 0;
    double  sum = 0.0;
    float   max = 0.0f;
};

// P(end of thinking) at a para or line position
struct common_reasoning_penalty_end_sample {
    int64_t tokens_in_think; // tokens accepted in reasoning blocks before this position
    char    cls;             // 'p' or 'l'
    float   p_end;
};

// P("\n") and P("\n\n") at a sent position
struct common_reasoning_penalty_nl_sample {
    int64_t tokens_in_think;
    float   p_single;
    float   p_double;
};

// max entries in each trace
#define COMMON_REASONING_PENALTY_TRACE_MAX 2000

struct common_reasoning_penalty_stats {
    int64_t triggers        = 0;
    int64_t markers         = 0;
    int64_t tokens_in_think = 0; // tokens accepted inside reasoning blocks, including the end tag token(s)

    std::vector<std::pair<std::string, int64_t>> hits; // per entry, only entries with hits

    // probability of the first end tag token, only for positions whose sampled token was accepted
    common_reasoning_penalty_p_stats                 p_end[COMMON_REASONING_PENALTY_POS_COUNT];
    std::vector<common_reasoning_penalty_end_sample> p_end_trace;

    // probability of "\n" / "\n\n" at sent positions
    int64_t                                         nl_n          = 0;
    double                                          nl_sum_single = 0.0;
    double                                          nl_sum_double = 0.0;
    std::vector<common_reasoning_penalty_nl_sample> sent_trace;
};

const char * common_reasoning_penalty_pos_name(enum common_reasoning_penalty_pos pos);

// the 50 marker words from the paper
std::vector<std::string> common_reasoning_penalty_default_words();

// Resolve words to single-token entries. Each word gives:
//   " " + word - always penalized, trigger if the word starts with an uppercase letter
//   word       - penalized only at line start, always a trigger
// An entry starting with '=' is a literal piece (escapes processed), not expanded and not gated.
// Spellings that do not tokenize to exactly one token are skipped and added to `dropped`.
// Each spelling is tokenized after a "\n" and the newline tokens are removed, so the SPM dummy space prefix
// does not turn "Wait" into " Wait". If the newline merges with the spelling, the plain tokenization is used.
std::vector<common_reasoning_penalty_entry> common_reasoning_penalty_resolve(
        const struct llama_vocab       * vocab,
        const std::vector<std::string> & words,
        std::vector<std::string>       * dropped);

// vocab is used for line start detection, position classes, the "\n" / "\n\n" tokens and stats labels, can be nullptr
// end_token is the first token of the reasoning end tag, its probability is tracked (LLAMA_TOKEN_NULL = off)
// the probabilities are computed from the logits as they arrive, before the penalty, and are committed to the stats
// only when the sampled token is accepted, so apply() without accept() (e.g. rejected draft positions) does not count
struct llama_sampler * common_reasoning_penalty_init(
        const struct llama_vocab                  * vocab,
        std::vector<common_reasoning_penalty_entry> entries,
        float                                       start,
        float                                       step,
        float                                       max,
        int32_t                                     window,
        llama_token                                 end_token = LLAMA_TOKEN_NULL);

// start a new reasoning block: reset n, the window and the line start state, keep the stats
void common_reasoning_penalty_new_block(struct llama_sampler * smpl);

// current value of n
int32_t common_reasoning_penalty_get_n(const struct llama_sampler * smpl);

// cumulative stats across all blocks
common_reasoning_penalty_stats common_reasoning_penalty_get_stats(const struct llama_sampler * smpl);
