
#이미지 처리를 위한 opencv
#얼굴인식을 위한 dlib
import cv2, dlib, sys
import numpy as np

scaler = 0.4

#얼굴 detector module 초기
detector = dlib.get_frontal_face_detector()

#얼굴 특징점 모듈 초기 
predictor = dlib.shape_predictor('face_detect_module/shape_predictor_68_face_landmarks.dat')

img_list = ['1.라이언', '2.아이언맨', '3.슈렉']
print(img_list)
choose = int(input("원하는 마스크 숫자 입력 :"))

# load overlay image
if choose == 1:
  overlay = cv2.imread('mp4_images/ryan_transparent.png', cv2.IMREAD_UNCHANGED)
elif choose == 2:
  overlay = cv2.imread('mp4_images/ironman.png', cv2.IMREAD_UNCHANGED)
elif choose == 3:
  overlay = cv2.imread('mp4_images/shrek.png', cv2.IMREAD_UNCHANGED)
  
#비디오 loading
cap = cv2.VideoCapture('mp4_images/child.mp4')


# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  bg_img = background_img.copy()
  # convert 3 channels to 4 channels
  if bg_img.shape[2] == 3:
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

  if overlay_size is not None:
    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

  b, g, r, a = cv2.split(img_to_overlay_t)

  mask = cv2.medianBlur(a, 5)

  h, w, _ = img_to_overlay_t.shape
  roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

  img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
  img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

  bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

  # convert 4 channels to 4 channels
  bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

  return bg_img

face_roi = []
face_sizes = []

# loop
while True:
  # 동영상을 프레임 단위로 읽기
  ret, img = cap.read()
  if not ret:
    break

  # 이미지 축소 (mp4)
  img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
  ori = img.copy()

  # find faces
  if len(face_roi) == 0:
    faces = detector(img, 1)
  else:
    roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
    # cv2.imshow('roi', roi_img)
    faces = detector(roi_img)

  # no faces
  if len(faces) == 0:
    print('face찾는중!')

  # find facial landmarks
  for face in faces:
    if len(face_roi) == 0:
      dlib_shape = predictor(img, face)
      shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
    else:
      dlib_shape = predictor(roi_img, face)
      shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])

    for s in shape_2d:
      cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 얼굴 중앙값
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

    # 얼굴 테두리
    min_coords = np.min(shape_2d, axis=0)
    max_coords = np.max(shape_2d, axis=0)

    # 좌표값 그리기
    cv2.circle(img, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    #face size 연산
    face_size = max(max_coords - min_coords)
    face_sizes.append(face_size)
    if len(face_sizes) > 10:
      del face_sizes[0]
    mean_face_size = int(np.mean(face_sizes) * 1.8)

    # 얼굴 관심영역 연산
    face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
    face_roi = np.clip(face_roi, 0, 10000)

    # draw overlay on face
    result = overlay_transparent(ori, overlay, center_x + 8, center_y - 25, overlay_size=(mean_face_size, mean_face_size))

  # visualize
  cv2.imshow('original', ori)
  cv2.imshow('facial landmarks', img)
  cv2.imshow('result', result)
  if cv2.waitKey(1) == ord('q'):
    sys.exit(1)
