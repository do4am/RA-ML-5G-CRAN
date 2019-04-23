function [classes_library_updated, classes_sample_idx_updated, log_vector] = cl_Merger(classes_library, classes, classes_idx, cl_numb_check)
    %Removing overlapping by assigning samples to bigger classes w.r.t the size
    %Before running, load S1-library.mat.
    %@INPUT:
    % classes_library. e.g., classes_library_updated.mat
    % classes. e.g., classes_sorted.mat  (sorted in descending order)
    % classes_idx. e.g., classes_sorted_idx (sorted in descending order) - CELL type
    % cl_numb_check. e.g., starting class. By default it is the first classes 
    %@OUTPUT:
    % classes_library_updated: classes_library will be updated in such a way unique assignment to one position
    % classes_sample_idx_updated: number of sample per class changes too, also reupdated
    % log_vector: log the change. 3 columns: classes'encoded name, original size, new size.
    %NOTE: sum of all new size should equal to the total number of samples.
  
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
        
        % count how many RAs to each sample in the considering class
        number_classes_per_sample_single = group_samples_in_class(classes_idx{i}(:,:), ... 
            number_classes_per_sample_all);
        
        % samples with multiple RAs are treated as nonexclusive
        nonexclusive_samples = find(number_classes_per_sample_single > 1);
        
        % log the change after non-exclusive samples are removed from class
        log_vector(i,:) = [i, size(number_classes_per_sample_single,1), ... 
            (size(number_classes_per_sample_single,1) - length(nonexclusive_samples))];
        fprintf('class: %d size: %d changed to: %d\n',log_vector(i,:));
        
        % update the class library. Removing class from non exlusive samples in class library.
        for j = 1:length(nonexclusive_samples)
            for jj = 1:50 %max columns of classes_library
                if classes_library_updated(classes_idx{i}(nonexclusive_samples(j)), jj) == classes(i,1)
                    classes_library_updated(classes_idx{i}(nonexclusive_samples(j)), jj) = 0;
                end
            end     
        end
        
         % count how many RAs to each samples in all classes
        number_classes_per_sample_all = number_classes_per_sample(classes_library_updated);
        classes_sample_idx_updated{i}(nonexclusive_samples,:) = [];
    end
