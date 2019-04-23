%get number of classes a sample belongs to; all samples of all classes
function [classes_library_updated, classes_sample_idx_updated, log_vector] = cl_merger(classes_library, classes, classes_idx, cl_numb_check)
    
    if nargin < 4
        cl_numb_check = 1;
    end
    
    number_classes_per_sample_all = number_classes_per_sample(classes_library);
    %Clustering
    classes_library_updated = classes_library;

    iterations = length(classes_idx);
    log_vector = zeros(iterations-cl_numb_check+1,3);
    classes_sample_idx_updated = classes_idx;

    for i = iterations:-1:cl_numb_check %start from the last classes

        number_classes_per_sample_single = group_samples_in_class(classes_idx{i}(:,:), ... 
            number_classes_per_sample_all);
        
        nonexclusive_samples = find(number_classes_per_sample_single > 1);
        log_vector(i,:) = [i, size(number_classes_per_sample_single,1), ... 
            (size(number_classes_per_sample_single,1) - length(nonexclusive_samples))];
        fprintf('class: %d size: %d changed to: %d\n',log_vector(i,:));
        
        for j = 1:length(nonexclusive_samples)
            for jj = 1:50 %max columns of classes_library
                if classes_library_updated(classes_idx{i}(nonexclusive_samples(j)), jj) == classes(i,1)
                    classes_library_updated(classes_idx{i}(nonexclusive_samples(j)), jj) = 0;
                end
            end     
        end
        
        number_classes_per_sample_all = number_classes_per_sample(classes_library_updated);
        classes_sample_idx_updated{i}(nonexclusive_samples,:) = [];
    end
