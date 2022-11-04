from scipy.ndimage import label
import torch

# TODO test this function when the first results are obtained
def post_process(prediction):
    wt = prediction[0]
    tc = prediction[1]
    et = prediction[2]
    
    result = torch.zeros_like(wt, dtype=torch.uint8)
    result[(wt >= 0.45 and tc < 0.4)] = 2
    result[(wt >= 0.45 and et < 0.45)] = 1
    result[(wt >= 0.45 and et >= 0.45)] = 4
    
    et = result == 4
    et_prob = prediction[2]
    
    label_comp, n_comp = label(et)
    for k in range(1, n_comp+1):
        kth_comp = label_comp == k
        elem_count = torch.count_nonzero(kth_comp) 
        mean_prob = torch.mean(et_prob[kth_comp])
        
        if elem_count < 16 and mean_prob < 0.9:
            result[kth_elem] = 1
    
    overall_elem_count = np.count_nonzero(et)
    overall_mean_prob = np.mean(et_prob[et])
    if overall_elem_count < 73 and overall_mean_prob < 0.9:
        result[et] = 1
