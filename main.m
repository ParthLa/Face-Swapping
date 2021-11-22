%% affine
clc
clear all
close all;

source = imread('Images/ex3.jpg');
target = imread('Images/ex1.jpg');

im_source = applyTransform(source,target);

target = double(target);
source = double(source);

%% mask
% creating a mask 
figure;
imshow(im_source, []);
im_source = double(im_source);
axis on; 
set(gcf,'Position',get(0,'Screensize')); 

h = imfreehand(); 
im_mask = h.createMask(); 

im_out = target; 

n = sum(sum(im_mask==1));  %pixels under mask 

laplacian_mask = [0 1 0;1 -4 1;0 1 0];

mapping = zeros(size(im_mask));

count = 0;

[sz1, sz2] = size(mapping);
for i=1:sz1
    for j=1:sz2
        if (im_mask(i,j) == 1)
            count = count + 1;
            mapping(i,j) = count;
        end
    end
end


% iterate for each channel
for channel = 1:3 
    xyv = zeros(5*n, 3);
    nz = 0;
    count =0; 

    b = zeros(n, 1);
    
    % taking only central part of the convolved image
    imglapl = conv2(im_source(:,:,channel),laplacian_mask, 'same');
    
    for k=1:sz1
        for l=1:size(mapping,2)
            if (im_mask(k,l)==1)
                count = count +1; 
                nz = nz + 1;
                xyv(nz, :) = [count, count, 4];
           
                for dx = -1:1
                    for dy = -1:1
                        if (dx*dx + dy * dy ~= 1)
                            continue
                        end
                        if (im_mask(k+dx, l+dy) == 0)
                            b(count) = b(count) + target(k+dx,l+dy,channel);
                        else
                            nz = nz + 1;
                            xyv(nz, :) = [count, mapping(k+dx,l+dy), -1];
                        end
                    end
                end
                b(count) = b(count) - imglapl(k,l); 
            end
        end
    end
    
    % sparse topelitz matrix to save space
    A = sparse(xyv(1:nz, 1), xyv(1:nz, 2), xyv(1:nz, 3), n, n, 5*n);
    x = A\b; 
  
    count = 0;
    for u=1:sz1
        for v=1:sz2
            if (im_mask(u,v) == 1)
                count = count + 1;
                im_out(u, v, channel) = x(count);
            end
        end
    end
end
figure; 

im_out = uint8(im_out);
imshow(im_out,[]); 

       
    




