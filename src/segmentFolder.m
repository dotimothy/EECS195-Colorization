% Segment Folder Implementation in Matlab %
function segmentFolder(sourceName,destName)
images = [dir(sourceName + "/*.JPG");dir(sourceName + "/*.png")];
for i = 1:length(images)
    sourcePath = sourceName + "\" + images(i).name;
    destPath = destName + "\" + images(i).name;
    tic
    segmentImage(sourcePath,destPath);
    toc
end
end
