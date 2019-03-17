%sort class sample size in descending order
function [sorted_classes, sorted_classes_sample_idx] = cl_sort(classes, classes_sample_idx)

    all_classes_size = zeros(length(classes_sample_idx),1);
    for i = 1:length(classes_sample_idx)
        all_classes_size(i) = numel(classes_sample_idx{i});   
    end
    
    [~, count_idx_sorted] = sort(all_classes_size,'descend');
    sorted_classes = classes(count_idx_sorted,:);
    sorted_classes_sample_idx = classes_sample_idx(count_idx_sorted,:);