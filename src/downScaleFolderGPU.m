% DownScale Folder Implementation in Matlab %
% GPU Optimization Assumes Images are the Same Dimension %
function downScaleFolderGPU(sourceName,destName)
images = [dir(sourceName + "/*.JPG");dir(sourceName + "/*.png")];
[r,c,d] = size(imread(sourceName + "\" + images(1).name));
blockSize = 1;
for k = 1:length(images)/blockSize
    set = zeros(r,c,d,blockSize);
    set = uint8(set); 
    % Reading All Images %
    for i = 1:blockSize
        sourcePath = sourceName + "\" + images(k*i).name;
        set(:,:,:,i) = imread(sourcePath);
    end
    % GPU Processing
    setG = gpuArray(set);
    output = imresize(setG,1080/c);
    for i = 1:blockSize
        destPath = destName + "\" + images(k*i).name;
        imwrite(output(:,:,:,i),destPath);
    end
end

end