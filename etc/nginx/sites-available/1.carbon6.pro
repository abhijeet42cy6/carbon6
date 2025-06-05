server {
    listen 80;
    server_name 1.carbon6.pro;

    location / {
        proxy_pass http://52.5.167.111:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
