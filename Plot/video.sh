ffmpeg -r 2 -f image2 -s 1920x1080 -i vel_u_fre_seq_%d.png -vcodec libx264 -pix_fmt yuv420p cylinder_fre.mp4
