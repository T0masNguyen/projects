t = timer('TimerFcn', 'stat=false; disp(''Timer!'')',... 
                 'StartDelay',3);
start(t)

stat=true;
while(stat==true)
  disp('.')
  pause(1)
end
disp('Schleife wurde beendet')
stop(t);