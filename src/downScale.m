% Downscale Implementation in Matlab %
function downScale(sourcePath,destPath)
    image = imread(sourcePath);
    output = imresize(image,[224 224]);
    imwrite(output,destPath);
end 