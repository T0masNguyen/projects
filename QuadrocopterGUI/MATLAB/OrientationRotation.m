
figure('Name', 'Serial communication: MATLAB + Arduino = CUBE 3D');
hold on
s = tic;
while toc(s) < 10
    x = randi(180);
    y = randi(180);
    z = randi(180);
    pause(0.05);
    orientation = readOrientation(r);
    Plot(x,y,z);
    drawnow limitrate;
    clf;
    
end