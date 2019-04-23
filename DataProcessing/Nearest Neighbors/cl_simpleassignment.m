function [V_lib, cl_size] = cl_simpleassignment(iV, V_lib, i_neighbors, cl_size)
%iV index of current Vertice
%V library
%i_neighbors idex of neighbors to current vertex, descending order of distance
%current size of all classes
%mode1 : assign to largest sample size class
%mode2 : force assignment

    n_idx = length(i_neighbors);
    class_name = {};

    Vi_classes_idx = find(V_lib(iV,:));
    Vi_classes = V_lib(iV, Vi_classes_idx); %all classes of the current vertice

    for i = 1:n_idx
        %export all the classes of the neighbors
        [~,~,class_name{i}] = find(V_lib(i_neighbors(i),:)); %#ok<AGROW>
    end 

    Vi_int_neighbors = intersect([class_name{:}], Vi_classes);  
    l = length(Vi_int_neighbors);
    
    if l >= 1
        V_lib(iV,Vi_classes_idx) = 0;
        V_lib(iV,(1:l)+1) = Vi_int_neighbors; 
        
        %update cl_size set  
        for i = 1:length(Vi_classes)
            cl_size(cl_size(:,1) == Vi_classes(i),2) = ...
                cl_size(cl_size(:,1) == Vi_classes(i),2) - 1;
        end 
        for i = 1:length(Vi_int_neighbors)
            cl_size(cl_size(:,1) == Vi_int_neighbors(i),2)  = ...
                 cl_size(cl_size(:,1) == Vi_int_neighbors(i),2) + 1;
        end   
    end 

