import torch
import torch.nn.functional as F


def get_scaled_dot_product(query_embs: torch.FloatTensor, passage_embs: torch.FloatTensor, temperature: int) -> torch.Tensor:
    return torch.matmul(query_embs, passage_embs.t()) * temperature


def calculate_cross_entropy_loss(similarity_scores: torch.Tensor) -> torch.Tensor:
    """
    This loss expects as input a square matrix, where we assume that (a_i, p_i) are a positive pair
    and (a_i, p_j) for i!=j are negative pairs.

    For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
    n-1 negative examples (p_j). We calculate the softmax normalized scores.

    E.g. the scores along the diagonal correspond to correct pairings (e.g. row=query, column=passage).
    Thus, all non-diagonal scores are incorrect pairings, and we can use them as in-batch negatives.
    Since diagonals are correct pairings, the labels are simply denoted by torch.arange(len(similarity_scores))

    functional.cross_entropy returns the negative log-softmax, i.e.: -mean(log(softmax(logits)))
    """
    return torch.nn.functional.cross_entropy(similarity_scores, torch.arange(len(similarity_scores), device=similarity_scores.device))


def get_nll_of_input_ids(log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    log_probs: this is the generation probability.
    - at each token: log-softmax of logits over the vocab_size
    - shape = [batch_size, max_seq_length - 1, vocab_size] , e.g. [2, 63, 32000]

    labels: this is generator input_ids, i.e. at each token, the (gold) subword-token-id
    - shape = [batch_size, max_seq_length - 1] , e.g. [2, 63]

    From the 'log_probs' tensor [2, 63, 32000], we just want the particular log-softmax-logits associated
    with the input_ids specified by 'labels' tensor. Thus we do:
    - labels.unsqueeze(2) -> [2, 63, 1]
    - now we can apply torch.apply on 'log_probs' -> [2, 63, 1]
    - squeeze(-1) -> discard the last dimension to derive [2, 63]

    Finally we return -ll (shape [2, 63]) , which is the negative-log-softmax(logits) of the (gold) subword-id,
    for the entire sequence of "query: <query> passage: <passage> answer: <answer>"
    """
    ll = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(-1)
    return -ll


def marginalize_log_probs(
    generator_logprobs: torch.FloatTensor, cosine_logprobs: torch.FloatTensor, query_token_length: torch.IntTensor
) -> torch.Tensor:
    """ NOT USED
    This function is called for each particular query
    logprobs_logits: next token generation probability for a particular
                     "query: <query> passage: <passage> answer: <answer>"
                     shape = (num# tokens, vocab size)
    doc_logprobs: log-softmax of cosine-sim of the particular <query> above, with the particular <passage> above
                  this is why we do the diag() operation above
                  shape = (1, 1)
    query_token_length: length of "query: <query> passage: <passage> answer:" (i.e. without <answer>)
                        in particular, the tokenizer breaks down #answer# into: [answer, :]
    """
    # we let the model to predict the next word for the query as is
    # Why is there a -1? Unless there is a end-of-sequence token that we are skipping over
    # logprobs_logits is the next-token generation probability from the causal LM. Since there is no "next token"
    # for the final token in the sequence, we take logits until (query_token_length - 1)
    query_passage_log_prob = generator_logprobs[: query_token_length - 1, :]

    # we marginalize the next word prediction of the passage based on the scores
    answer_log_prob = generator_logprobs[query_token_length - 1 :, :]

    marginalized_prob_sum = answer_log_prob + cosine_logprobs

    # get all the log probs
    all_log_probs = torch.cat([query_passage_log_prob, marginalized_prob_sum], dim=0)

    return all_log_probs


def calculate_marginalized_generator_loss(
    generator_logits: torch.LongTensor,  # [batch_size, generator_max_seq_length, vocab_size]
    input_ids: torch.Tensor,  # generator_input_input_ids ; shape=[batch_size, max_seq_len]
    attention_mask: torch.Tensor,  # generator_input_attention_mask ; shape=[batch_size, max_seq_len]
    cosine_logits: torch.Tensor,  # cosine similarity logits of (query, passage) ; shape=(num# query, num# passage), and we assume query and passage come in pairs
    query_passage_token_length: torch.Tensor,  # generator tokenized "query: <query> passage: <passage> answer:" ; shape=1d tensor
) -> torch.Tensor:
    """ NOT USED
    """
    # Here, generator_logits is the output from a AutoModelForCausalLM, where the LM is providing
    # next-token probabiliity for: "query: <query> passage: <passage> answer: <answer>"
    # generator_logits.shape = (num# batch query, num# tokens, vocab size)
    # AutoRegressive models like AutoModelForCausalLM generate the probability distribution for the next token
    # given the context of the previous tokens. For these models, the last token in a sequence is often used
    # as the input to predict the next token, but since there is no "next" token after the last one,
    # it doesn't make sense to compute a probability distribution for it. Thus, the -1
    generator_logprobs = F.log_softmax(generator_logits[:, :-1, :], dim=2).view(generator_logits.shape[0], -1, generator_logits.size(-1))
    # shape = [batch_size, generation max_seq_length -1, vocab_size]

    # Here I am assuming that we always take the positive sample as the correct one
    # Below, cosine_logits is the matrix of cosine similarity between query embeddings and passage embeddings.
    # 1. We convert the cosine similarity scores into log probabilities. The resulting tensor has the same shape
    #    as scores, and each row represents a distribution of log probabilities over passages for a particular query.
    # 2. cosine_logits is the cosine sim of (query, passage) where we assume that we feed 1-to-1 (query, passage)
    #    The diagonal represents the cosine sim of each query, to its respective correct passage
    cosine_logprobs = torch.log_softmax(cosine_logits, dim=1).diag().unsqueeze(-1).unsqueeze(-1)  # shape = [num# query, 1, 1]

    marginalized_next_word_prob_list = []
    for eg_generator_logprobs, eg_cosine_logprobs, eg_token_length in zip(
        # eg_generator_logprobs: next token generation probability for a particular
        #                         "query: <query> passage: <passage> answer: <answer>"
        #                         shape = (num# tokens, vocab size)
        # eg_cosine_logprobs: cosine similarity of the particular <query> above, with the particular <passage> above
        #                      this is why we do the diag() operation above
        #                      shape = (1, 1)
        # eg_token_length: length of "query: <query> passage: <passage> answer:"
        generator_logprobs, cosine_logprobs, query_passage_token_length
    ):
        # add the loss from cosine-sim to the answer part of the generation loss
        marginalized_log_probs = marginalize_log_probs(eg_generator_logprobs, eg_cosine_logprobs, eg_token_length)
        marginalized_next_word_prob_list.append(marginalized_log_probs)

    marginalized_log_probs = torch.stack(marginalized_next_word_prob_list)

    # we just want the negative log-likelihood (of the softmax over logits) for the gold_sub-words/input_ids
    loss = get_nll_of_input_ids(marginalized_log_probs, input_ids[:, 1:])  # input_tensors : generator_input_input_ids
    loss_tensor = loss * attention_mask[:, 1:]  # pay attention starting from the 2nd sub-word

    overall_average_loss = loss_tensor.sum() / attention_mask[:, 1:].sum()  # average loss over each sub-word token

    return overall_average_loss


def calculate_generator_loss(
        generator_logits: torch.LongTensor,  # [batch_size, generator_max_seq_length, vocab_size]
        input_ids: torch.Tensor,  # generator_input_input_ids ; shape=[batch_size, max_seq_len]
        attention_mask: torch.Tensor,  # generator_input_attention_mask ; shape=[batch_size, max_seq_len]
) -> torch.Tensor:
    # Here, generator_logits is the output from a AutoModelForCausalLM, where the LM is providing
    # next-token probabiliity for: "query: <query> passage: <passage> answer: <answer>"
    # generator_logits.shape = (num# batch query, num# tokens, vocab size)
    # AutoRegressive models like AutoModelForCausalLM generate the probability distribution for the next token
    # given the context of the previous tokens. For these models, the last token in a sequence is often used
    # as the input to predict the next token, but since there is no "next" token after the last one,
    # it doesn't make sense to compute a probability distribution for it. Thus, the -1
    generator_logprobs = F.log_softmax(generator_logits[:, :-1, :], dim=2).view(generator_logits.shape[0], -1,
                                                                                generator_logits.size(-1))
    # shape = [batch_size, generation max_seq_length -1, vocab_size]

    # we just want the negative log-likelihood (of the softmax over logits) for the gold_sub-words/input_ids
    # loss = get_nll_of_input_ids(marginalized_log_probs, input_ids[:, 1:])  # input_tensors : generator_input_input_ids
    loss = get_nll_of_input_ids(generator_logprobs, input_ids[:, 1:])
    loss_tensor = loss * attention_mask[:, 1:]  # pay attention starting from the 2nd sub-word

    # print(f"attention_mask[:, 1:]={attention_mask[:, 1:]}")
    overall_average_loss = loss_tensor.sum() / attention_mask[:, 1:].sum()  # average loss over each sub-word token

    return overall_average_loss
