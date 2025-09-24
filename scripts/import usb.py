import usb.core, usb.util

for dev in usb.core.find(find_all=True):
    # 0x0e (14) is "video" class, 0xEF (239) is "misc" (UVC webcams sometimes report this)
    if dev.bDeviceClass in (14, 239):
        try:
            name = usb.util.get_string(dev, dev.iProduct)
        except Exception:
            name = "Unknown"
        print(f"Bus {dev.bus}, Device {dev.address}, Class {dev.bDeviceClass}, Name: {name}")
