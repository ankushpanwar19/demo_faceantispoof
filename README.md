# demo_faceantispoof

sudo cp /path/to/your/index.html /var/www/html/
sudo vim /etc/nginx/sites-available/my_site
sudo ln -s /etc/nginx/sites-available/my_site /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx