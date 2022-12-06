serving = true;

while serving
    socket = tcpip('localhost', 50000, 'NetworkRole', 'server', 'Timeout', 5); %%#ok<TNMLP>
    fopen(socket);

    while socket.BytesAvailable < 1
        pause(0.01);
    end

    bytes = fread(socket, socket.BytesAvailable);
    action = char(bytes)'; 
    fprintf([action '\n']);

    if strcmp(action,'GETPARAMS')
        fwrite(socket,'x=1,y=2');
    elseif strcmp(action,'RUNSIM')
        fwrite(socket,'state=(a=1,b=2)');
    elseif strcmp(action,'EXIT')
        fwrite(socket,'exiting');
        serving = false;
    else
        fwrite(socket,'ERROR');
    end
    
    fclose(socket);       
end

