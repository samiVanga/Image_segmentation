import cv2
import numpy as np
import random
from tqdm import tqdm
from pipeline import process_images
from pipeline import apply_split_segmentation
from pipeline import apply_segmentation
from Threshold import threshold
from watershed import watershed_segmentation

from preprocessing import preprocess
from Threshold import threshold
from pipeline import process_images
from preprocessing import thresh_pre
from pipeline import input_image
from pipeline import apply_segmentation
from postProcessing import postProcessing
from evaluation import output
from evaluation import mean_intersection




class Region:
#references the code from the labs

    def __init__(self, coords, slice):

        self.coords = coords
        self.slice = slice
        self.mean = np.mean(self.slice)
        self.std = np.std(slice)
        self.edges = []
        self.index = None

    def split(self):


        def adjacent(bbox1, bbox2):

            xmin_1, ymin_1, xmax_1, ymax_1 = bbox1
            xmin_2, ymin_2, xmax_2, ymax_2 = bbox2
            return xmin_1 == xmax_2 or xmax_1 == xmin_2 or ymin_1 == ymax_2 or ymax_1 == ymin_2

        def connectSubregion2Neighbour(subregion):


            for edge in self.edges:

                neighbour = edge.queryNeighbouor(self)
                bbox1, bbox2 = neighbour.returnBbox(), subregion.returnBbox()
                if adjacent(bbox1, bbox2):
                    new_edge = Edge(neighbour, subregion)
                    neighbour.edges.append(new_edge)
                    subregion.edges.append(new_edge)

        height, width = self.slice.shape
        meshgrid = self.coords.reshape(2, height, width)
        coord_left_top = meshgrid[:, :height // 2, :width // 2].reshape(2, -1)
        coord_right_top = meshgrid[:, :height // 2, width // 2:].reshape(2, -1)
        coord_left_bottom = meshgrid[:, height // 2:, :width // 2].reshape(2, -1)
        coord_right_bottom = meshgrid[:, height // 2:, width // 2:].reshape(2, -1)

        slice_left_top = self.slice[:height // 2, :width // 2]
        slice_right_top = self.slice[:height // 2, width // 2:]
        slice_left_bottom = self.slice[height // 2:, :width // 2]
        slice_right_bottom = self.slice[height // 2:, width // 2:]

        left_top = Region(coord_left_top, slice_left_top)
        right_top = Region(coord_right_top, slice_right_top)
        left_bottom = Region(coord_left_bottom, slice_left_bottom)
        right_bottom = Region(coord_right_bottom, slice_right_bottom)

        left_top2left_bottom = Edge(left_top, left_bottom)
        right_top2right_bottom = Edge(right_top, right_bottom)
        left_top2right_top = Edge(left_top, right_top)
        left_bottom2right_bottom = Edge(left_bottom, right_bottom)

        left_top.edges = [left_top2right_top, left_top2left_bottom]
        right_top.edges = [left_top2right_top, right_top2right_bottom]
        left_bottom.edges = [left_bottom2right_bottom, left_top2left_bottom]
        right_bottom.edges = [left_bottom2right_bottom, right_top2right_bottom]

        connectSubregion2Neighbour(left_top)
        connectSubregion2Neighbour(right_top)
        connectSubregion2Neighbour(left_bottom)
        connectSubregion2Neighbour(right_bottom)

        for edge in self.edges:

            neighbour = edge.queryNeighbouor(self)

            neighbour.edges.remove(edge)

        return left_top, right_top, left_bottom, right_bottom

    def returnBbox(self):

        x_min = self.coords[0, 0]
        y_min = self.coords[1, 0]
        x_max = self.coords[0, -1] + 1
        y_max = self.coords[1, -1] + 1
        return x_min, y_min, x_max, y_max


class Edge:

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.weight = 0

        self.calculateWeight()

    def calculateWeight(self):

        self.weight = abs(self.left.mean - self.right.mean)

    def queryNeighbouor(self, query):

        return self.right if query == self.left else self.left


class SplitMergeMaster:

    def __init__(self, image, split_thresh=14, merge_thresh=17):

        self.image = image.copy()
        self.split_thresh = split_thresh
        self.merge_thresh = merge_thresh

        self.meshgrid = np.stack(np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0])), axis=0).reshape(2, -1)

        self.regions = []
        self.edges = []

    def _split(self):

        regions = [Region(self.meshgrid, self.image)]

        while len(regions) != 0:

            region = regions.pop()
            if region.std >= self.split_thresh and region.slice.shape[0] > 1 and region.slice.shape[1] > 1:
                left_top, right_top, left_bottom, right_bottom = region.split()
                regions.extend([left_top, right_top, left_bottom, right_bottom])
            else:

                region.index = len(self.regions)
                self.regions.append(region)
        for region in self.regions:
            self.edges.extend(region.edges)
        self.edges = list(set(self.edges))

    def _merge2regions(self, region_1, region_2, current_edge):

        region_1.coords = np.hstack([region_1.coords, region_2.coords])
        region_1.mean = (np.sum(self.image[region_1.coords[1, :], region_1.coords[0, :]]) + np.sum(self.image[region_2.coords[1, :], region_2.coords[0, :]])) / (region_1.coords.shape[1] + region_2.coords.shape[1])
        for edge in region_2.edges:
            if edge.left == region_2:
                edge.left = region_1
            if edge.right == region_2:
                edge.right = region_1
            edge.calculateWeight()
        region_1.edges.extend(region_2.edges)
        region_1.edges = list(set(region_1.edges))
        self.regions[region_2.index] = None

    def segmentation(self):

        self._split()

        while True:
            self.edges.sort(key=lambda x: x.weight)
            more_change = True
            for edge in self.edges:
                if edge.left != edge.right and edge.weight <= self.merge_thresh:
                    more_change = False
                    self._merge2regions(edge.left, edge.right, edge)
                    break
            if more_change:
                break

        masks = []
        for region in self.regions:
            if region is not None:
                mask = np.zeros(self.image.shape, dtype=np.uint8)
                mask[region.coords[1, :], region.coords[0, :]] = 1
                masks.append(mask)
        masks.sort(reverse=True, key=lambda x: np.sum(x))
        return np.stack(masks, axis=0)

    @staticmethod
    def visualization(masks):

        colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue

        ]


        visual_map = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)

        for i in range(masks.shape[0]):
            color_index = i % len(colors)
            visual_map[masks[i] == 1] = colors[color_index]

        # Calculate the middle color
        center_y, center_x = visual_map.shape[1] // 2, visual_map.shape[2] // 2
        center_pixel_color = visual_map[center_y, center_x]

        final_visual_map = np.zeros_like(visual_map)

        for i in range(3):
            if np.array_equal(center_pixel_color,colors[i]):
                final_visual_map[masks[i] == 1] = [0, 0, 0]
            else:
                final_visual_map[masks[i] == 1] = [255, 255, 255]

        return final_visual_map


if __name__ == "__main__":
    # read the image
    gt = process_images("ground_truths", "gt_images", input_image)
    i_image = process_images("Images", "input-images", input_image)
    process_images("Images", "preprocessing-images", preprocess)
    apply_split_segmentation("preprocessing-images", "split-middle-output",SplitMergeMaster)
    mask = process_images("split-middle-output", "split-Output", postProcessing)
    output(i_image, mask)
    mean_intersection(mask, gt)