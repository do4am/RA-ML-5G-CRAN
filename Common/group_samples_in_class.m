%function 
function number_classes_total = group_samples_in_class(class_i, number_classes)   
    iter_pos = size(class_i,1);
    number_classes_total = zeros(iter_pos,1);
    pos_idx = class_i(:,1);

    for j = 1:iter_pos
        number_classes_total(j) = number_classes(pos_idx(j)); 
    end
