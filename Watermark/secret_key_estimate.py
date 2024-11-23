import torch
from transformers import AutoTokenizer
import numpy as np
import random
from watermark import MyWatermarkedModel, query_model, verifier


def get_predicted_intervals(input_str, model, tokenizer, max_new_tokens):
    inputs = tokenizer(input_str, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, 
                             return_dict_in_generate=True, output_scores=True)
    predicted_tokens = outputs.sequences.squeeze()[-max_new_tokens:]
    empirical_intervals = []
    for i in range(max_new_tokens):
        scores_processed = outputs.scores[i].clone().flatten().softmax(dim=-1)
        cumul_scores = scores_processed.cumsum(dim=-1).squeeze()
        empirical_interval = (cumul_scores[predicted_tokens[i] - 1].item(), 
                              cumul_scores[predicted_tokens[i]].item())
        empirical_intervals.append(empirical_interval)
    return empirical_intervals

def find_max_overlapping_interval(intervals):
    '''
    Empirically estimate the maximum overlapping interval.

    Args:
        intervals: with shape (sample_nums, max_new_tokens, 2), representing 
            two positive boundaries of the interval of predicted token.
    Return:
        max_overlap: mean of the maximum overlapping interval 
            if there is no overlap, return mean of all intervals
    '''
    intervals = np.array(intervals)
    sample_nums, max_new_tokens, _ = intervals.shape
    overlap_means = []
    for i in range(max_new_tokens):
        start = intervals[:, i, 0]
        end = intervals[:, i, 1]
        overlap_mean = (min(start) + max(end)) / 2
        overlap_means.append(overlap_mean)
    return overlap_means

def main():
    MAX_NEW_TOKENS = 4
    model_path = './watermarked_model.pt'
    model = torch.load(model_path)
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilgpt2')

    # add more random prompts here
    prompts = [
        "Once upon a time",
        "Hello, my dog is cute", 
        "Good morning, my",
        "The quick brown fox jumps over the lazy dog",
        "How are you doing today",
        "Hello my name",
        "This is a test",
        "Thanks for coming and",
        "I am so happy to",
        "The weather is so nice today",
        "I am going to the",
        "He is a very good",
        "She is a very good",
    ]

    # get empirical estimated interval for r[0:3]
    pred_intervals = []
    for input_str in prompts:
        pred_interval = get_predicted_intervals(input_str, model, tokenizer, 
                                                 MAX_NEW_TOKENS)
        pred_intervals.append(pred_interval)
    max_overlaps = find_max_overlapping_interval(pred_intervals)
    np.save('rs.npy', max_overlaps[:3])


if __name__ == '__main__':
    main()