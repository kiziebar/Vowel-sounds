for i=1:10
    for gloska = ['a','e','i','o','u','y']
        recObj = audiorecorder(44100, 16, 1);
        fprintf('Mow gloske: %s',gloska)
        pause(1.5)
        recordblocking(recObj, 5);
        y = getaudiodata(recObj);
        
        filename = sprintf('dzwiek_%s_%d.wav',gloska,i);
        
        audiowrite(filename,y,44100);
        clc
    end
    pause(10)
end