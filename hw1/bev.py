import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        ### TODO ###
        W, H, depth, focal = 512, 512, 2.5, 256 # focal is computed from fov, and WH
        roll = -(np.pi / 2)
        cos_val = np.cos(roll)
        if np.isclose(cos_val, 0, atol=1e-10):
            cos_val = 0.0
        sin_val = np.sin(roll)
        if np.isclose(sin_val, 0, atol=1e-10):
            sin_val = 0.0
        
        top_homo_c2w = np.eye(4)
        top_homo_c2w[:3, :3] = np.array([[1, 0, 0], [0, cos_val, -sin_val], [0, sin_val, cos_val]]) # rotation
        top_homo_c2w[:3, 3] = top_T = np.array([0, -2.5, 0]) # translation
        
        front_homo_c2w = np.eye(4)
        front_homo_c2w[:3, 3] = np.array([0, -1, 0])
        front_homo_w2c = np.linalg.inv(front_homo_c2w)
        
        c2c = np.matmul(front_homo_w2c, top_homo_c2w)
        

        
        uv = np.array(points)
        # top_image_coord to top_camera_coord
        x = (uv[:, 0:1] - W*.5) * depth / focal
        y = (uv[:, 1:] - H*.5) * depth / focal
        z = np.full(x.shape, depth)
        xyz = np.concatenate([x, y, z, np.ones(x.shape)], -1)
        xyz = xyz[:, :, np.newaxis] # homogeneous
        # top_camera_coord to front_camera_coord
        c2c = np.tile(c2c, (uv.shape[0], 1, 1))
        xyz = np.matmul(c2c, xyz)
        # front_camera_coord to front_image_coord
        u = xyz[:, 0] / xyz[:, 2] * focal + W*.5
        v = xyz[:, 1] / xyz[:, 2] * focal + H*.5
        new_pixels = np.round(np.concatenate([u, v], -1)).astype(int).tolist()

        print("new_pixels", new_pixels)
        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90

    front_rgb = "bev_data/front1.png"
    top_rgb = "bev_data/bev1.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)
