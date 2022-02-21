% Downscale Implementation in Matlab %
function downScale(sourcePath,destPath)
    image = imread(sourcePath);
    output = imresize(image,1080/size(image,2));
    imwrite(output,destPath);
end 