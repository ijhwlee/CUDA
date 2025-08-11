#!/bin/bash

while IFS=: read -r username grpname home_dir shell pwd; do
    sudo useradd -m -G "$grpname" -d "$home_dir" -s "$shell" "$username"
    echo "$username:$pwd" | sudo chpasswd
    sudo passwd --expire "$username"
    echo "User $username added with home $home_dir and shell $shell"
done < users.txt

#sudo deluser --remove-home username
