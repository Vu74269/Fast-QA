import torch

def predict(tokenizer, model, question, context, null_score_diff_threshold=1.0):
    # Tokenize in the same order as preprocessing: question, context.
    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=384,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )

    # keep and remove offsets before calling model (model doesn't accept offset_mapping kwarg)
    offset_mapping = inputs.pop("offset_mapping", None)

    # Move tensors to model device (exclude offset_mapping)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0].cpu()
    end_logits = outputs.end_logits[0].cpu()

    input_ids = inputs["input_ids"][0].cpu().tolist()
    token_type_ids = inputs.get("token_type_ids", inputs.get("token_type_ids", None))
    if token_type_ids is not None:
        token_type_ids = token_type_ids[0].cpu().tolist() if hasattr(token_type_ids[0], 'cpu') else token_type_ids[0]
    else:
        # if token_type_ids missing, create a mask of zeros (treat as all context False)
        token_type_ids = [0] * len(input_ids)

    # restore offsets from saved mapping
    if offset_mapping is None:
        offsets = [[0, 0] for _ in range(len(input_ids))]
    else:
        if isinstance(offset_mapping, torch.Tensor):
            offsets = offset_mapping[0].cpu().tolist()
        else:
            # already a list
            offsets = offset_mapping[0]

    # find CLS index
    try:
        cls_index = input_ids.index(tokenizer.cls_token_id)
    except ValueError:
        cls_index = 0

    # best token indices
    start_idx = int(torch.argmax(start_logits).item())
    end_idx = int(torch.argmax(end_logits).item())

    # scores
    best_span_score = float((start_logits[start_idx] + end_logits[end_idx]).item())
    no_answer_score = float((start_logits[cls_index] + end_logits[cls_index]).item())

    # approximate normalized confidence
    start_probs = torch.softmax(start_logits, dim=0)
    end_probs = torch.softmax(end_logits, dim=0)
    prob_score = float((start_probs[start_idx] * end_probs[end_idx]).item())

    # decide no-answer using same threshold as notebook
    if no_answer_score > best_span_score + null_score_diff_threshold:
        return {"answer": "", 
                "score": prob_score, 
                "debug": {"best_span_score": best_span_score, 
                          "no_answer_score": no_answer_score, 
                          "start_idx": start_idx, 
                          "end_idx": end_idx}}

    # ensure indices in right order
    if end_idx < start_idx:
        end_idx = start_idx

    # ensure predicted tokens are inside context (token_type_id == 1)
    if token_type_ids[start_idx] != 1 or token_type_ids[end_idx] != 1:
        return {"answer": "", 
                "score": prob_score, 
                "debug": {"best_span_score": best_span_score, 
                          "no_answer_score": no_answer_score, 
                          "start_idx": start_idx, 
                          "end_idx": end_idx}}

    # map token indices to character positions in context using offsets
    start_char = offsets[start_idx][0]
    end_char = offsets[end_idx][1]

    answer = context[start_char:end_char].strip()
    if answer == "":
        return {"answer": "", 
                "score": prob_score, 
                "debug": {"best_span_score": best_span_score, 
                          "no_answer_score": no_answer_score, 
                          "start_idx": start_idx, 
                          "end_idx": end_idx}}

    return {"answer": answer, 
            "score": prob_score, 
            "debug": {"best_span_score": best_span_score, 
                      "no_answer_score": no_answer_score, 
                      "start_idx": start_idx, 
                      "end_idx": end_idx}}