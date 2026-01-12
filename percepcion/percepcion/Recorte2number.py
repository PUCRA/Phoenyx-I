import joblib
import os
import cv2
import numpy as np



class Recorte2number():
    def __init__(self):
        # Path to the trained KNN model (expanded to the user's home directory)
        ruta = os.path.expanduser("~/phoenyx_ws/src/percepcion/percepcion/final_knn.pkl")
        # Load the trained KNN classifier
        self.knn = joblib.load(ruta)

        # Temporal memory to avoid bounding box flickering between frames
        self.prev_bbox = None
        # Temporal smoothing factor (higher = more stable, lower = more responsive)
        self.alpha = 0.7   # temporal smoothing coefficient

    # ============================================================================================
    #                                   KNN
    # ============================================================================================
    def obtener_knn_num(self, img_thresh):
        # Resize the binary image to the size expected by the KNN
        img_resized = cv2.resize(img_thresh, (50, 50), interpolation=cv2.INTER_AREA)
        # Flatten the image into a 1D feature vector
        img_flat = img_resized.reshape(1, -1)
        # Predict and return the digit using the trained KNN
        return self.knn.predict(img_flat)[0]

    # ============================================================================================
    #                                   MAIN PIPELINE
    # ============================================================================================
    def obtener_colorYnum(self, image):
        try:
            # Validate input image
            if image is None or image.size == 0:
                return None, "Indefinido", None

            # ==================================================
            # COLOR DETECTION
            # ==================================================
            # Convert image from BGR to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define HSV range for blue color
            lower_blue = np.array([100, 150, 50])
            upper_blue = np.array([140, 255, 255])

            # Define HSV ranges for red color (two ranges due to hue wrap-around)
            lower_red1 = np.array([0, 150, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 150, 50])
            upper_red2 = np.array([180, 255, 255])

            # Create binary masks for blue and red colors
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            mask_red = (
                cv2.inRange(hsv, lower_red1, upper_red1) +
                cv2.inRange(hsv, lower_red2, upper_red2)
            )

            # Count how many pixels belong to each color
            blue_pixels = cv2.countNonZero(mask_blue)
            red_pixels = cv2.countNonZero(mask_red)

            # Decide the dominant color and corresponding mask
            if blue_pixels > red_pixels:
            # Count how many pixels belong to each color
                color = "Azul"
                mask = mask_blue
            elif red_pixels > 0:
                color = "Rojo"
                mask = mask_red
            else:
                # No relevant color detected
                return None, "Indefinido", None

            # ==================================================
            # CARD (COLORED BACKGROUND) DETECTION
            # ==================================================
            # Find contours of the detected color mask
            cont_cart, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cont_cart:
                return None, color, None

            # Select the largest contour as the colored card
            cartulina = max(cont_cart, key=cv2.contourArea)
            # Compute bounding box of the card
            x, y, w, h = cv2.boundingRect(cartulina)
            # Extract region of interest (ROI) containing the card
            roi = image[y:y+h, x:x+w]

            # ==================================================
            # PREPROCESSING
            # ==================================================
            # Convert ROI to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Remove a small border to avoid edge artifacts
            pad = max(int(min(w, h) * 0.05), 5)
            gray = gray[pad:-pad, pad:-pad]

            # Validate cropped grayscale image
            if gray.size == 0:
                return None, color, None

            # --- CORRECT THRESHOLDING (NOT INVERTED) ---
            # Apply Otsu's binarization
            _, img_thresh = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Light morphological cleaning to remove noise
            kernel = np.ones((3, 3), np.uint8)
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            # ==================================================
            # NUMBER BOUNDING BOX
            # ==================================================
            # Extract a normalized bounding box around the digit
            numero_img = self.bounding_box(img_thresh)
            if numero_img is None:
                return None, color, None

            # ==================================================
            # FILL NUMBER WHILE PRESERVING HOLES (e.g., 0, 6, 8, 9)
            # ==================================================
            # Find contours and hierarchy to distinguish holes
            contours, hierarchy = cv2.findContours(
                numero_img,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours or hierarchy is None:
                return None, color, None

            # Create an empty image to draw the filled number
            img_filled = np.zeros_like(numero_img)

            hierarchy = hierarchy[0]

            for i, cnt in enumerate(contours):
                # External contour (no parent)
                if hierarchy[i][3] == -1:
                    cv2.drawContours(img_filled, [cnt], -1, 255, thickness=cv2.FILLED)
                # Internal contour (hole)
                else:
                    cv2.drawContours(img_filled, [cnt], -1, 0, thickness=cv2.FILLED)

            # ==================================================
            # NORMALIZATION FOR KNN
            # ==================================================
            # Resize the final digit image to KNN input size
            img_final = cv2.resize(img_filled, (50, 50))

            # Reject empty images
            if cv2.countNonZero(img_final) == 0:
                return None, color, None

            # ==================================================
            # KNN CLASSIFICATION
            # ==================================================
            # Predict the digit using the KNN model
            numero = self.obtener_knn_num(img_final)

            # ==================================================
            # SMOOTHED BOUNDING BOX DRAWING (VISUAL FEEDBACK)
            # ==================================================
            # Find contours again for visualization
            cont_num, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cont_num:
                cont = max(cont_num, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(cont)

                # Current bounding box in global image coordinates
                curr_bbox = (x + pad + bx, y + pad + by, bw, bh)

                # Initialize or smooth bounding box position
                if self.prev_bbox is None:
                    self.prev_bbox = curr_bbox
                else:
                    self.prev_bbox = tuple(
                        int(self.alpha * p + (1 - self.alpha) * c)
                        for p, c in zip(self.prev_bbox, curr_bbox)
                    )

                # Draw the smoothed bounding box
                px, py, pw, ph = self.prev_bbox
                cv2.rectangle(image, (px, py), (px + pw, py + ph), (0, 255, 0), 2)

            # ==================================================
            # DEBUG VISUALIZATION
            # ==================================================
            # Show the final image that is fed into the KNN, used when debuging
            # cv2.imshow("Imagen que entra a la KNN (rellena)", img_final)
            # cv2.waitKey(1)

            # Return detected number, color, and processed digit image
            return numero, color, img_final

        except Exception as e:
            # Catch and report any unexpected error
            print(f"[ERROR] obtener_colorYnum: {e}")
            return None, "Indefinido", None


    # ============================================================================================
    #                                   ROTATED BOUNDING BOX
    # ============================================================================================
    def ordenar_puntos_bounding_box(self, puntos):
        # Order points of a rotated rectangle consistently (top-left, top-right, bottom-right, bottom-left)
        s = puntos.sum(axis=1)
        d = np.diff(puntos, axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = puntos[np.argmin(s)]
        rect[2] = puntos[np.argmax(s)]
        rect[1] = puntos[np.argmin(d)]
        rect[3] = puntos[np.argmax(d)]
        return rect

    def bounding_box(self, binaria):
        # Find contours in the binary image
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return None

        # Select the largest contour
        cont = max(contornos, key=cv2.contourArea)
        # Discard very small contours (noise)
        if cv2.contourArea(cont) < 200:
            return None

        # Compute minimum area rotated rectangle
        rect = cv2.minAreaRect(cont)
        centro, size, angulo = rect
        # Use the largest side to form a square
        lado = int(max(size))
        if lado <= 0:
            return None

        # Get the four corner points of the rotated rectangle
        box = cv2.boxPoints((centro, (lado, lado), angulo))
        box = np.intp(box)

        # Define destination square coordinates
        destino = np.array([
            [0, 0],
            [lado - 1, 0],
            [lado - 1, lado - 1],
            [0, lado - 1]
        ], dtype="float32")

        # Order source points and compute perspective transform
        origen = self.ordenar_puntos_bounding_box(np.float32(box))
        M = cv2.getPerspectiveTransform(origen, destino)
        # Warp the binary image to obtain a straightened digit
        warped = cv2.warpPerspective(binaria, M, (lado, lado))

        # Normalize size and apply final smoothing
        warped = cv2.resize(warped, (50, 50))
        return self.suavizar_numero(warped)

    # ============================================================================================
    #                                   FINAL SMOOTHING
    # ============================================================================================
    def suavizar_numero(self, img):
        # Find contours and hierarchy to preserve holes
        contornos, jer = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # Output image initialization
        out = np.zeros_like(img)

        if jer is None:
            return img

        holes = []
        for i, c in enumerate(contornos):
            # Approximate contour for smoother shape
            eps = 0.01 * cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, eps, True)
            # External contour
            if jer[0][i][3] == -1:
                cv2.drawContours(out, [c], -1, 255, cv2.FILLED)
            # Internal contour (hole)
            else:
                holes.append(c)

        # Draw holes in black
        for h in holes:
            cv2.drawContours(out, [h], -1, 0, cv2.FILLED)

        # Return the smoothed digit image
        return out
