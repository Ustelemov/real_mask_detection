# if __name__ == '__main__':
#     # stream = urllib.request.urlopen('http://10.117.223.205:555/tkZlcetY?container=mjpeg&stream=main')
#     # bytes = bytes()

# frame = cv2.imread("input.jpg")
# image = image_processor.read_image_rgb_float(frame)
# input_image, scale, pad = image_processor.image_pad_and_scale(image)
# input_image = np.transpose(input_image,[2,0,1])[np.newaxis,:,:,:]
# # model forward
# time1 = time.time()
# predict_x = model.forward(input_image)
# print(time.time()-time1)
# # post process
# humans = post_processor.process(predict_x)[0]
# # visualize results (restore detected humans)
# print(f"{len(humans)} humans detected")
# for human_idx,human in enumerate(humans,start=1):
#     human.unpad(pad)
#     human.unscale(scale)
#     print(f"human:{human_idx} num of detected body joints:{human.get_partnum()}")
#     human.print()

# cv2.imshow("1",visualize_on_image(image=frame, humans=humans, name="result"))
# cv2.waitKey(1000*1000)



if __name__ == '__main__':
    # stream = urllib.request.urlopen('http://10.117.223.205:555/tkZlcetY?container=mjpeg&stream=main')
    # bytes = bytes()

    cap = cv2.VideoCapture('socgum1.mp4')

    while True:
        # bytes += stream.read(1024)
        # a = bytes.find(b'\xff\xd8')
        # b = bytes.find(b'\xff\xd9')
        # if a != -1 and b != -1:
        #     jpg = bytes[a:b+2]
        #     bytes = bytes[b+2:]
        #     frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        ret,frame = cap.read()
        frame = cv2.resize(frame, (1280,720))
        frame = frame[:440,150:550]


        image = image_processor.read_image_rgb_float(frame)
        input_image, scale, pad = image_processor.image_pad_and_scale(image)
        input_image = np.transpose(input_image,[2,0,1])[np.newaxis,:,:,:]
        # model forward
        time1 = time.time()
        predict_x = model.forward(input_image)
        print(time.time()-time1)
        # post process
        humans = post_processor.process(predict_x)[0]
        # visualize results (restore detected humans)
        print(f"{len(humans)} humans detected")
        for human_idx,human in enumerate(humans,start=1):
            human.unpad(pad)
            human.unscale(scale)
            print(f"human:{human_idx} num of detected body joints:{human.get_partnum()}")
        
        frame = visualize_on_image(image=frame, humans=humans, name="result")

        # # vis_image = visualize(frame, faces)
        cv2.imshow('Video', frame)

        key = cv2.waitKey(1) & 0xFF

        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()







