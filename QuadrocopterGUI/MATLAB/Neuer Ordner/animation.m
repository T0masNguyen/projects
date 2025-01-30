r = app.tello;
ani.figure = open('triad.fig');
ani.triad = [ani.figure.Children.Children];

hx = line([0,1.1],[0,0],[0,0],'color',[0.5,0,0],'LineStyle','--');
hy = line([0,0],[0,-1.1],[0,0],'color',[0,0.5,0],'LineStyle','--');
hz = line([0,0],[0,0],[0,-1.1],'color',[0,0,0.5],'LineStyle','--');
ani.triadRef = [hx,hy,hz];
legend([hx,hy,hz],'Roll','Pitch','Yaw');
xlim([-1 1]*1.1)
ylim([-1 1]*1.1)
zlim([-1 1]*1.1)
dim = [.65 .1 .1 .1];
x = 0;
y = 0;
z = 0;
str = "Yaw: " + x + " Pitch: " + y + " Roll: " + z;
text = annotation('textbox',dim,'String',str,'FitBoxToText','on');
axis on


%angle of rotation
previousx = 0;
previousy = 0;
previousz = 0;


while true
    try 
    rad = rad2deg(readOrientation(r));
    if isempty(rad) == false
    x = rad(3);
    y = rad(2);
    z = rad(1);  
    %x-axis = roll - red
    %y-axis = pitch - green
    %z-axis = yaw - blue
    str = "Yaw: " + z + " Pitch: " + y + " Roll: " + x;
    set(text, 'String',str);
    
    newx = x - previousx;
    rotate(ani.triad,[1 0 0], newx);
    previousx = x;

    newy = -(y - previousy); 
    rotate(ani.triad,[0 1 0], newy);
    previousy = y;
    
    newz = -(z - previousz);
    rotate(ani.triad,[0 0 1], newz);
    previousz = z;
    
    end
    
    catch
        clear r;
        break;
    end
end



	

