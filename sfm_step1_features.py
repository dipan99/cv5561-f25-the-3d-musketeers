import os
import glob
import cv2
import numpy as np


def load_images(image_dir, exts=(".jpg", ".jpeg", ".png")):
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
    image_paths = sorted(image_paths)
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 images, found {len(image_paths)} in {image_dir}")
    images = [cv2.imread(p) for p in image_paths]
    for i, img in enumerate(images):
        if img is None:
            raise ValueError(f"Failed to load image: {image_paths[i]}")
    return image_paths, images


def to_gray(images):
    gray_images = []
    for img in images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray_images.append(gray)
    return gray_images


def create_feature_detector():
    # You can switch to SIFT if available:
    # return cv2.SIFT_create()
    return cv2.ORB_create(nfeatures=4000)


def detect_and_describe(detector, gray_images):
    keypoints_list = []
    descriptors_list = []
    for idx, img in enumerate(gray_images):
        kps, des = detector.detectAndCompute(img, None)
        if des is None or len(kps) == 0:
            raise RuntimeError(f"No features found in image index {idx}")
        keypoints_list.append(kps)
        descriptors_list.append(des)
        print(f"[detect] Image {idx}: {len(kps)} keypoints")
    return keypoints_list, descriptors_list


def match_descriptors(des1, des2, ratio=0.75):
    # For ORB (binary descriptors) use Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def match_all_pairs(descriptors_list):
    num_images = len(descriptors_list)
    pair_matches = {}  # (i, j) -> list of cv2.DMatch
    for i in range(num_images - 1):
        j = i + 1
        des1 = descriptors_list[i]
        des2 = descriptors_list[j]
        good_matches = match_descriptors(des1, des2)
        pair_matches[(i, j)] = good_matches
        print(f"[match] Pair ({i}, {j}): {len(good_matches)} good matches")
    return pair_matches


def visualize_matches(images, keypoints_list, pair_matches):
    """Visualize consecutive pairs - press any key to go to next pair"""
    for (i, j), matches in sorted(pair_matches.items()):
        # Draw matches between image i and j
        img_matches = cv2.drawMatches(
            images[i], keypoints_list[i],
            images[j], keypoints_list[j],
            matches[:100],  # Show top 100 matches
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Add text label with larger font
        label = f"Pair ({i}, {j}): {len(matches)} matches - Press any key for next"
        cv2.putText(img_matches, label, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Resize to be larger and clearer - max width 1920
        max_width = 1920
        if img_matches.shape[1] < max_width:
            scale = max_width / img_matches.shape[1]
            new_width = max_width
            new_height = int(img_matches.shape[0] * scale)
            img_matches = cv2.resize(img_matches, (new_width, new_height))
        
        # Show each pair in full screen
        cv2.namedWindow(f"Match Pair ({i}, {j})", cv2.WINDOW_NORMAL)
        cv2.imshow(f"Match Pair ({i}, {j})", img_matches)
        print(f"[visualize] Showing pair ({i}, {j}). Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # TODO: set this to your folder with room images
    image_dir = "./images"  # change to your path

    print("[main] Loading images...")
    image_paths, images = load_images(image_dir)
    print(f"[main] Loaded {len(images)} images")

    print("[main] Converting to grayscale...")
    gray_images = to_gray(images)

    print("[main] Creating feature detector...")
    detector = create_feature_detector()

    print("[main] Detecting features...")
    keypoints_list, descriptors_list = detect_and_describe(detector, gray_images)

    print("[main] Matching consecutive image pairs...")
    pair_matches = match_all_pairs(descriptors_list)

    print("[main] Done. Summary:")
    for (i, j), matches in pair_matches.items():
        print(f"  Pair ({i}, {j}): {len(matches)} matches")
    
    # Save features and matches for next step
    print("[main] Saving features and matches...")
    np.savez("./step1_features.npz",
             image_paths=image_paths,
             keypoints=[np.array([kp.pt for kp in kps]) for kps in keypoints_list],
             descriptors=descriptors_list,
             pair_matches={(i, j): [(m.queryIdx, m.trainIdx, m.distance) 
                          for m in matches] for (i, j), matches in pair_matches.items()})
    print("[main] Features saved to step1_features.npz")
    
    print("[main] Visualizing matches...")
    visualize_matches(images, keypoints_list, pair_matches)

    #     # Choose which pair to visualize
    # pair_to_show = (3, 4)  # change this, for example (0, 1) or (3, 4)

    # if pair_to_show in pair_matches:
    #     i, j = pair_to_show
    #     matches = pair_matches[(i, j)]
    #     print(f"[main] Visualizing matches for pair ({i}, {j})...")
    #     draw_matches(
    #         images[i], keypoints_list[i],
    #         images[j], keypoints_list[j],
    #         matches,
    #         max_to_show=50
    #     )
    # else:
    #     print(f"[main] Pair {pair_to_show} not found in pair_matches.")

def draw_matches(img1, kp1, img2, kp2, matches, max_to_show=50):
    # Take only first N matches so image isn't cluttered
    matches_to_draw = matches[:max_to_show]
    out = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("Matches", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
