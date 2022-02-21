% DownScale Folder Implementation in Matlab %
function downScaleFolder(sourceName,destName)
images = [dir(sourceName + "/*.JPG");dir(sourceName + "/*.png")];
for i = 1:length(images)
    sourcePath = sourceName + "\" + images(i).name;
    destPath = destName + "\" + images(i).name;
    downScale(sourcePath,destPath);
end
end