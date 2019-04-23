function number_classes_per_samples_for_all = number_classes_per_sample(classes_library)
% conunt how many RAs to each samples for all dataset
% @INPUT: 
% classes_library
% @OUTPUT:
% number_classes_per_samples_for_all: all classes from c_1 ---> c_T

    number_classes_per_samples_for_all = zeros(length(classes_library),1);
    for i = 1:length(classes_library)
        number_classes_per_samples_for_all(i) = numel(find(classes_library(i,:)>0));   
    end