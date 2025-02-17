# Deploy FASTAPI on AWS EC2
For the full stack code, please refer to: https://github.com/Jiang-Feiyu/Development-of-a-Full-stack-Aphasia-AI-Web-App-based-on-React-FASTAPI 

### Launch an EC2 Instance
After register an AWS account, go to `Console Home` and click to `EC2 Dashboard`. Config an EC2 Instance as following:
```
Name
Machine Image: Ubuntu (Latest long term supported version)
Instance Type: t2.micro 
```

- Create a `Key pair` to login by ssh
```
Key pair name
Key pair type: RSA
Private key format: .pem
```
Once you've created a key, you will see a key downloaded.

- Network Settings
```
Allow ssh traffic form: anywhere
Allow http access from the internet
Allow https access from the internet
```

**Launch instance!** Go back to the EC2 dashboard and you will see your instance is running like:

<img src="https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f79094a188e249d19117055a97f4edc4~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1102&h=79&s=27269&e=png&b=fdfdfd" alt="螢幕截圖 2024-03-27 上午11.42.12.png" width="80%" />

Information like public IP can be viewed:

<img src="https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/acd4871a60eb4cacb1652392f04c1cde~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1111&h=557&s=125379&e=png&b=fefefe" alt="螢幕截圖 2024-03-27 上午11.45.33.png" width="80%" />

### Connect to the instance
Login by SSH client (Mac or Linux)
```
chomd 400 [name].pem
```
Then just copy the ssh command.Remember you need to go to the directory where your pem file stored before paste the SSH command on your terminal. You will see this after login:

<img src="https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ca101cc514b24ec8bdcec738d3278688~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=948&h=873&s=135153&e=png&b=010101" alt="螢幕截圖 2024-03-27 上午11.50.48.png" width="50%" />

For easeier SSH connection, may congif your ssh in your `.ssh` dir as following:
        
```
Host aws
        Hostname [your server].compute.amazonaws.com
        User ubuntu
        IdentityFile ~/.ssh/fast_api_key.pem
```
And directly use `ssh aws` to login.

### Server settings
```
sudo apt install -y python3-pip nginx
```

Create configuration file:
```
sudo vim /etc/nginx/sites-enabled/fastapi_nginx

server {
    listen 80;   
    server_name 18.116.199.161;    
    location / {        
        proxy_pass http://127.0.0.1:8000;    
    }
}

sudo service nginx restart
```
Then clone your FASTAPI project to your server.

Install all the dependencies and packages by:
```
pip3 install -r requirement.txt
```

### Start the FASTAPI
```
python3 -m uvicorn --host 0.0.0.0  main:app
```

### Increase EC2 storage
I found the 8 GiB is not enough for my project so I decided to upgrade the storage.
- Click on EC2 instance and then go to the storage session
<img src="https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8cbf731e1e8f44ab83df0253c90bb4a5~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1115&h=507&s=86116&e=png&b=fefefe" alt="螢幕截圖 2024-03-27 下午12.08.14.png" width="70%" />

- Then modify the volume:
<img src="https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8ad33f72150244d8975ea6b8fb1ceb05~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=846&h=627&s=70250&e=png&b=fefefe" alt="螢幕截圖 2024-03-27 下午12.09.08.png" width="70%" />

- So far we have increase the size of EC2, we need to cofig our server by ssh login
- List block devices, `svda1` is your original hard drive
    ```
    lsblk
    ```
    <img src="https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/18e0f3e4502b4a5c86b759dc04f0c35c~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=807&h=360&s=79719&e=png&b=020202" alt="螢幕截圖 2024-03-27 下午12.14.22.png" width="50%" />

- Increase our partition
    ```
    sudo growpart /dev/xvda 1
    ```
    You can see the volume increse by `lsblk`
- Extend our file system
    ```
    sudo resize2fs /dev/xvda1
    ```
    You should see the volume got extend
    
    <img src="https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4567860ba8274aaf8b44c9051358a270~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=692&h=239&s=55670&e=png&b=020202" alt="螢幕截圖 2024-03-27 下午12.21.12.png" width="50%" />

### Security Setting
If new port needed (default 80), you may consider to add a rule:
```
TCP [port] 0.0.0.0/0(all ports)
```
<img src="https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/edf72349579e4ef5830cff9bab6a561f~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=2490&h=938&s=172556&e=png&b=ffffff" alt="螢幕截圖 2024-03-27 下午5.50.05.png" width="70%" />

### Test connection
```
netcat [ip address] [port]
curl http://[ip address]:[port]
```
A common problem is the Chorme may automatically set `http`->`https`, whcih may result in `Invalid HTTP request received`. One way to solve this is to deploy the frontend to the server as well. 

Another problem may encounter is `Mixed Content: This request has been blocked; the content must be servered over HTTPS`. You can add
```
<meta http-equiv="Content-Security-Policy" content="upgrade-insecure-requests">
```
at the haed part of your `public/index.html` of your React front-end.

### Visit from browser
Only Chrome from laptop is allowed to visit the web page.
 
- Browser settings
        Type the URL of `chrome://flags/#unsafely-treat-insecure-origin-as-secure` in your chrome and give the permission of http://3.27.151.169:3000 , change the permission to Enabled and then restart the Chrome. 
- You can login with an initial account:
  ```
        Username: user1@hku.hk
        Password: ^Jz2kBu2%To1
  ```
- Audios
        You can test the Audio Recognition function by uploading the audio or just record it online, the estimated response time is 2-4 seconds. Test audios could be found [Here] 

### `Tmux`
A session manager that runs on Unix. One of the key features of Tmux is its ability to run in the background, which means you can keep your terminal sessions running even after you close the terminal window or disconnect from the remote server. This makes it useful for long-running tasks or for working on remote machines.

Useful cmds:
```
tmux -> create a new session
Ctrl+b, d -> quit the session
tmux list-sessions -> list sessions
tmux attach-session -t 1 -> go to the exist session
```

### Google Private Mode
`Shift command N`

### Other useful links:
[How to Deploy FastAPI on AWS EC2: Quick and Easy Steps]

[here]: https://github.com/Jiang-Feiyu/FYP-Full-stack-Aphasia-AI-Web-App-Development-based-on-React-and-FASTAPI

[How to Deploy FastAPI on AWS EC2: Quick and Easy Steps]:https://www.youtube.com/watch?v=SgSnz7kW-Ko

[Here]: https://github.com/Jiang-Feiyu/FYP-Full-stack-Aphasia-AI-Web-App-Development-based-on-React-and-FASTAPI/tree/main/testAudio
