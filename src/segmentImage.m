% Segment Implementation in Matlab %
% Gets Average of Superpixels %
function segmentImage(sourceName,destName)
    image = imread(sourceName);
    [L,N] = superpixels(image,10000);
    outputImage = zeros(size(image),'like',image);
    idx = label2idx(L);
    numRows = size(image,1);
    numCols = size(image,2);
    for labelVal = 1:N
        redIdx = idx{labelVal};
        greenIdx = idx{labelVal}+numRows*numCols;
        blueIdx = idx{labelVal}+2*numRows*numCols;
        outputImage(redIdx) = mean(image(redIdx));
        outputImage(greenIdx) = mean(image(greenIdx));
        outputImage(blueIdx) = mean(image(blueIdx));
    end   
    imwrite(outputImage,destName);
end
