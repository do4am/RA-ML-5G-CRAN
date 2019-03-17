function [V_lib, cl_size] = cl_assignment(iV, V_lib, i_neighbors, cl_size)
%iV index of current Vertice
%V library
%i_neighbors idex of neighbors to current vertex
%current size of all classes
    
    n_idx = length(i_neighbors);
    class_name = {};
    
    Vi_classes_idx = find(V_lib(iV,:));
    Vi_classes = V_lib(iV, Vi_classes_idx);
    V_lib(iV,Vi_classes_idx) = 0;
     
    for i = 1:n_idx
        %export all the classes of the neighbors
        [~,~,class_name{i}] = find(V_lib(i_neighbors(i),:)); %#ok<AGROW>
    end 
    
    all_name = [class_name{:}];
    [~,f_commons,commons] = mode(all_name); %returned value commons always a cell
    n_common = size(commons{:},1); % # of RAs that are dominant
    
    %Assignment alogirthm
    if n_common == 1   
        V_lib(iV,2) = commons{:};  %set value to any colums of lib does not matter         
    
    elseif f_commons > 1 %at least a common class between neighbors
        [~,~,commons_Vi] = mode([Vi_classes repmat(commons{:}',1,f_commons)]);
        if size(commons_Vi{:},1) == 1 %a class in Vi is in common class
            V_lib(iV,2) = commons_Vi{:};            
        
        else % 
            min_cl_size = [0 0];
            for i = 1:n_common
                current_cl_size = cl_size(cl_size(:,1) == commons{:}(i),2);
                if min_cl_size(1) < current_cl_size %assign to larger class
                    min_cl_size = [current_cl_size i];
                end
            end
            V_lib(iV,2) = commons{:}(min_cl_size(2));
        
        end
    else %no common RAs between neighbors
        %closest neighbor
        min_cl_size = [0 0];
        for i = 1:length(class_name{1,1}(:))
            current_cl_size = cl_size(cl_size(:,1) == class_name{1,1}(i),2);
            if min_cl_size(1) < current_cl_size
                min_cl_size = [current_cl_size i];
            end
        end 
        
        V_lib(iV,2) = class_name{1,1}(min_cl_size(2));
    
    end
    
    %update cl_size set        
    for i = 1:length(Vi_classes)
        cl_size(cl_size(:,1) == Vi_classes(i),2) = ...
            cl_size(cl_size(:,1) == Vi_classes(i),2) - 1;
    end    
    cl_size(cl_size(:,1) == V_lib(iV,2),2)  = ...
         cl_size(cl_size(:,1) == V_lib(iV,2),2)  + 1;
    %common_class_neighbor = intersect(class_name{:},Vi_classes
    %
    
   