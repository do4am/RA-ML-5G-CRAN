%how many class 1 positions belong to
function samples_size_in_all_classes = number_classes_per_sample(current_class_clustering)
    samples_size_in_all_classes = zeros(length(current_class_clustering),1);
    for i = 1:length(current_class_clustering)
        samples_size_in_all_classes(i) = numel(find(current_class_clustering(i,:)>0));   
    end