#!/bin/bash -uxe

brew upgrade cmake

case ${CUDA:0:3} in

'7.5')  installer="http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.27_mac.dmg";;
'8.0')  installer="https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_mac-dmg";;
'9.0')  installer="https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_mac-dmg";;
'9.1')  installer="https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.128_mac";;
'9.2')  installer="https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda_9.2.64_mac";;

esac

wget -O cuda.dmg "$installer"

brew install p7zip
7z x cuda.dmg

brew install gnu-tar
sudo gtar -x --skip-old-files -f CUDAMacOSXInstaller/CUDAMacOSXInstaller.app/Contents/Resources/payload/cuda_mac_installer_tk.tar.gz -C /
