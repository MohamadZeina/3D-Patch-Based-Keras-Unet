import os
import re
import time
import socket
import random
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from datetime import datetime
from keras.utils import Sequence
from keras.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler

class PatchSequence(Sequence):
    """Takes a list of paths to niftis, and returns them in batches of patches"""

    def __init__(self, x_lists, y_lists, batch_size,
                 patch_size, stride, volumes_to_analyse, first_vol=0, unet=True,
                 validation=False, secondary_input=True, spatial_offset=0,
                 randomise_spatial_offset=False, reslice_isotropic=0, x_only=False,
                 prediction=False, shuffle=True, verbose=True, BraTS_like_y=False,
                 flip_gradients=False, slice_volume=(0, 0,  0, 0,  0, 0)):

        # Load arguments into class variables
        # todo: add a try except here, which catches non-lists fed in
        # makes them a list of 1
        self.no_of_xs = len(x_lists)
        self.no_of_ys = len(y_lists)

        # Otherwise, zipping a list of length 1 (one patient with many modalities)
        # will zip along the wrong axis and leave unusable paths
        self.one_patient_only = type(x_lists[0]) == str or type(x_lists[0]) == np.str_

        if self.one_patient_only:
            self.x_lists = x_lists
            self.y_lists = y_lists
            #print("One patient loaded: \nx: %s \ny:%s" %(self.x_lists, self.y_lists))
        else:
            self.x_lists = list(zip(*x_lists))[first_vol: first_vol + volumes_to_analyse]
            self.y_lists = list(zip(*y_lists))[first_vol: first_vol + volumes_to_analyse]

        if not self.one_patient_only:
            combined = list(zip(self.x_lists, self.y_lists))
            random.shuffle(combined)
            self.x_lists[:], self.y_lists[:] = zip(*combined)

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride
        self.unet = unet

        self.volumes_to_analyse = volumes_to_analyse
        self.validation = validation
        self.secondary_input = secondary_input
        self.spatial_offset = spatial_offset
        self.randomise_spatial_offset = randomise_spatial_offset
        self.reslice_isotropic = reslice_isotropic
        self.x_only = x_only
        self.prediction = prediction
        self.shuffle = shuffle
        self.verbose = verbose
        self.BraTS_like_y = BraTS_like_y
        self.flip_gradients = flip_gradients
        self.slice_volume = slice_volume

        # Ensures a volume is loaded on first request
        self.x_volume_needed = True
        self.y_volume_needed = True

        # Ensures the same image is not analysed multiple times consecutively
        self.next_x = 0
        self.next_y = 0

        # This ensures correct indexing for patches after the first volume
        self.x_offset = 0
        self.y_offset = 0

        # To benchmark the patch extraction part of the code
        self.time_patching = []

        # Make important calculations, necessary for len method
        self.patches_x = {}
        self.patches_y = {}

        # Gets the number of patches from each volume
        self.total_patches = 0
        self.patches_each_volume = []
        for i in range(self.volumes_to_analyse):
            if self.reslice_isotropic != 0:

                # I have not applied correction here for padding.
                #  Reslice isotropic isn't currently being used and just
                #  here for legacy support

                (patches_i,
                 patches_j,
                 patches_k,
                 patches) = self.get_total_patches(self.reslice_isotropic,
                                                   self.reslice_isotropic,
                                                   self.reslice_isotropic)
            else:

                # Deals with weird errors from single patient, 
                #  multiple modalities
                if self.one_patient_only:
                    image = nib.load(self.x_lists[0])
                else:
                    image = nib.load(self.x_lists[i][0])


                dims = image.header["dim"][1:4]
                # dims = dims - self.spatial_offset

                # Correct for the padding which will be applied later, so
                #  the volume divides into patches exactly
                odd_bits = np.mod(dims, self.patch_size)
                dims_to_pad = [patch_size - odd_bit if odd_bit > 20 else 0
                               for odd_bit in odd_bits]
                dims = dims + dims_to_pad

                #print("estimated dims from header info are: ", dims)

                (patches_i,
                 patches_j,
                 patches_k,
                 patches) = self.get_total_patches(dims[0], dims[1], dims[2])

            self.total_patches += patches
            self.patches_each_volume.append(patches)

        # This helps UnetEvaluator reconstruct the patches into the correct arrangement
        self.patch_arrangement = [patches_i, patches_j, patches_k]
        #print("Estimated patch arrangement from header: ", self.patch_arrangement)

        self.batches_per_volume = patches // int(batch_size)
        self.total_batches = int(np.ceil(self.total_patches / int(batch_size)))

        # Print summary
        if self.verbose:
            print("\n" + "#" * 8 + " New generator intialised with the following "
                                   "properties: " + "#" * 8)
            print("This generator will produce patches from a volume of "
                  "dimensions (after padding): ", dims)
            print("These should correspond to a patch_arrangement of ", self.patch_arrangement)
            print("There should be approximately %i patches for each volume"
                  % patches)
            print("This should lead to approximately %i batches per volume"
                  % self.batches_per_volume)
            print("The total number of batches will be ", self.total_batches)
            print("#" * 8 + " End new generator properties " + "#" * 8 + "\n")

    def __len__(self):
        return self.total_batches

    def get_total_patches(self, dims_i, dims_j, dims_k, only_total=False):
        """Given the 3 dimensions of an image, and the patch size and stride
        already specified in the instance, this will give back how many
        steps will be taken in each of the dimensions"""

        # I have no confidence in my maths, but I've checked this empirically
        patches_i = int(
            np.floor((dims_i - (self.patch_size)) / self.stride) + 1)
        patches_j = int(
            np.floor((dims_j - (self.patch_size)) / self.stride) + 1)
        patches_k = int(
            np.floor((dims_k - (self.patch_size)) / self.stride) + 1)

        total_patches = patches_i * patches_j * patches_k

        if only_total:
            return total_patches
        else:
            return patches_i, patches_j, patches_k, total_patches

    def get_linear_gradients(self, dims):

        grad_i = np.zeros(dims, dtype=np.float)
        grad_j = np.zeros(dims, dtype=np.float)
        grad_k = np.zeros(dims, dtype=np.float)

        for i in range(dims[0]):
            grad_i[i, :, :] = i / dims[0]

        for j in range(dims[1]):
            grad_j[:, j, :] = j / dims[1]

        for k in range(dims[2]):
            grad_k[:, :, k] = k / dims[2]

        linear_gradients = [grad_i, grad_j, grad_k]
        linear_gradients = [self.spatial_offset_func(grad) for grad in linear_gradients]
        linear_gradients = np.moveaxis(linear_gradients, 0, 3)

        # This is in case you trained on brains in one orientation, but wish to test
        #  on brains in a mirrored orientation
        if self.flip_gradients:
            linear_gradients = np.flip(linear_gradients, 1)

        return linear_gradients

    def reslice(self, volume):

        if self.reslice_isotropic == 0:
            return volume

        original_dims = np.array(volume.shape)

        # Perform reslicing
        volume = zoom(volume, (self.reslice_isotropic / original_dims))

        return volume

    def spatial_offset_func(self, volume):
        """Crops the volume by self.spatial_offset, the pads it in
        the other direction to preserve dimensions"""

        if self.spatial_offset == 0:
            return volume

        volume = volume[
                 self.spatial_offset:, self.spatial_offset:, self.spatial_offset:]

        volume = np.pad(volume,
                        ((0, self.spatial_offset),
                         (0, self.spatial_offset),
                         (0, self.spatial_offset)),
                        'edge')

        return volume

    def load_volume(self, path):

        volume = nib.load(path).get_data()

        volume = self.spatial_offset_func(volume)
        volume = self.reslice(volume)

        #if self.slice_volume == (0, 0,  0, 0,  0, 0):
        #    volume = volume
        #else:
        #    volume = volume[self.slice_volume[0]:self.slice_volume[1],
        #                  self.slice_volume[2]:self.slice_volume[3],
        #                  self.slice_volume[4]:self.slice_volume[5]]

        dims = volume.shape
        leeway = 20

        # Calculate dimensions in each axis that are not patched, and pad
        odd_bits = np.mod(dims, self.patch_size)
        # print("Dimensions not patched: ", odd_bits)
        # Doesn't pad when odd bit is 0
        dims_to_pad = [self.patch_size - odd_bit if odd_bit > leeway else 0
                       for odd_bit in odd_bits]

        # Save this so it can be used by e.g. UnetEvaluator to see how much things were padded
        self.dims_to_pad = dims_to_pad

        #print("Dims to pad are ", dims_to_pad)

        # print("Therefore, padding should take dimensions: ", dims_to_pad)
        volume = np.pad(volume,
                        ((dims_to_pad[0] // 2, -(-dims_to_pad[0] // 2)),
                         (dims_to_pad[1] // 2, -(-dims_to_pad[1] // 2)),
                         (dims_to_pad[2] // 2, -(-dims_to_pad[2] // 2))),
                        'minimum')
        dims = volume.shape

        #print("Real dims after padding: ", dims)

        (self.iterations_i,
         self.iterations_j,
         self.iterations_k,
         self.total_indices) = self.get_total_patches(dims[0],
                                                      dims[1],
                                                      dims[2])
        true_patch_arrangement = [self.iterations_i, self.iterations_j, 
                                  self.iterations_k]

        #print("Real patch arrangement: ", true_patch_arrangement)
        total_patches = self.total_indices



        return volume

    def scale(self, volume):

        # Fit a scaler to the current image
        dims = volume.shape
        scaler = StandardScaler()
        volume = volume.reshape(-1, 1)
        volume = scaler.fit_transform(volume)
        volume = volume.reshape(dims[0], dims[1], dims[2])
        return volume

    class Patcher():
        """Will take an equivalent patch from any supplied volume

        Example:

        patcher = Patcher([i, j, k], self.patch_size)
        t1_patch = patcher.patch(t1_volume)
        seg_patch = patcher.patch(seg)

        """

        def __init__(self, indices, patch_size):
            self.idcs = indices
            self.patch_size = patch_size

        def patch(self, volume):
            patch = volume[self.idcs[0]: self.idcs[0] + self.patch_size,
                    self.idcs[1]: self.idcs[1] + self.patch_size,
                    self.idcs[2]: self.idcs[2] + self.patch_size, ]

            return patch

    def patch_from_volume_legacy(self, index, only_centre):
        """Legacy because I'm working on one which returns x and y at the
        same time. Given an index, it will find and return the corresponding
        single patch"""

        if self.x_volume_needed:
            self.x_volumes = [self.load_volume(path)
                              for path in self.x_lists[self.next_x]]
            self.x_volumes = [self.scale(volume) for volume in self.x_volumes]

            # Get linear gradients if secondary input require
            if self.secondary_input:
                self.linear_gradients = self.get_linear_gradients(
                    self.x_volumes[0].shape)

            self.x_volume_needed = False

            if self.shuffle:
                random.seed(42)
                # Create a mapping between requested and retrieved index, if shuffling
                requested_indices = list(range(self.total_indices))
                shuffled_indices = random.sample(requested_indices, self.total_indices)
                mapping_dict = {i: shuffled_indices[i] for i in requested_indices}
                reverse_mapping_dict = {shuffled_indices[i]: i
                                        for i in requested_indices}

                self.mapping_dict = mapping_dict
                self.reverse_mapping_dict = reverse_mapping_dict

        elif self.y_volume_needed:

            self.y_volumes = [self.load_volume(path)
                              for path in self.y_lists[self.next_y]]

            self.y_volume_needed = False

            if self.BraTS_like_y:
                # Process BraTS like segmentations, ie those with one channel
                #  and multiple values in that channel. Assumes only one volume
                volume = self.y_volumes[0]
                self.y_volumes = []

                for i in np.unique(volume):

                    # Skip the air class, which is added later anyway
                    if i == 0:
                        continue

                    self.y_volumes.append(volume == i)

        if only_centre:
            # First, apply the offsets to the index and then shuffle it
            unshuffled_index = index - self.y_offset
            if self.shuffle:
                shuffled_index = self.mapping_dict[unshuffled_index]
            else:
                shuffled_index = unshuffled_index

            offset = self.y_offset
            vol_dims = self.y_volumes[0].shape

            volumes = self.y_volumes


        else:
            # First, apply the offsets to the index and then shuffle it
            unshuffled_index = index - self.x_offset
            if self.shuffle:
                shuffled_index = self.mapping_dict[unshuffled_index]
            else:
                shuffled_index = unshuffled_index

            offset = self.x_offset
            vol_dims = self.x_volumes[0].shape

            volumes = self.x_volumes

        # Figuring out the corresponding patch for the requested index
        ij_area = self.iterations_i * self.iterations_j
        in_plane_index = shuffled_index % ij_area

        # This is the number of steps to take in each direction
        i_no = in_plane_index % self.iterations_i
        j_no = in_plane_index // self.iterations_i
        k_no = shuffled_index // ij_area

        # Convert from which position (i.e. 5th patch) to absolute location
        i = i_no * self.stride
        j = j_no * self.stride
        k = k_no * self.stride

        patcher = self.Patcher([i, j, k], self.patch_size)

        # Extract the patches
        if only_centre:
            # Extract patches for each of the 3 segmentation types
            # start_time = time.time()

            # Return either a patch, or a single voxel, depending on if
            #  unet is requested
            if self.unet:
                patches = [patcher.patch(volume) for volume in volumes]
                patches = np.moveaxis(patches, 0, 3)
                # print("Shape of y patches", patches.shape)

                # Background (air) class
                no_brain = np.ones_like(patches[:, :, :, 0]) - np.sum(patches, axis=3)
                no_brain = np.expand_dims(no_brain, -1)

                # print("shape of no brain patch after creating", no_brain.shape)
                patch = np.concatenate([patches, no_brain], axis=3)
                # print("shape of y patch after concat with air class is", patch.shape)

                # print("shape of y patch after moving axis is", patch.shape)

            else:
                patch = volume[int(i + self.patch_size / 2),
                               int(j + self.patch_size / 2),
                               int(k + self.patch_size / 2)]

                patch_2 = volume_2[int(i + self.patch_size / 2),
                                   int(j + self.patch_size / 2),
                                   int(k + self.patch_size / 2)]

                patch_3 = volume_3[int(i + self.patch_size / 2),
                                   int(j + self.patch_size / 2),
                                   int(k + self.patch_size / 2)]

                no_brain = 1 - (patch + patch_2 + patch_3)

                patch = np.array([patch, patch_2, patch_3, no_brain])

            # elapsed_time = time.time() - start_time
            # self.time_patching.append(elapsed_time)


        else:
            patch = [patcher.patch(volume) for volume in volumes]
            # patch = volume[i: i + self.patch_size,
            #        j: j + self.patch_size,
            #        k: k + self.patch_size]

            # print(volume.shape, i, j, k, self.patch_size)
            # print("size of x patch just extracted is ", patch.size)

            # to do: why do I do this now? Why don't I just concatenate it
            #  right when it's produced. Maybe the above code breaks
            #  if it ends with ..., :]? Could also be because I want to
            #  return two separate neural network inputs
            # To do: improve variable naming, it's confusing to switching
            #  names back to euclidean_distance halfway through
            if self.secondary_input:
                lin_grad_patch = patcher.patch(self.linear_gradients)
                # lin_grad_patch = self.linear_gradients[i: i + self.patch_size,
                #                 j: j + self.patch_size,
                #                 k: k + self.patch_size,
                #                 :]

                # Calculate the euclidean distance to give cnn spatial awareness
                # mid_point = np.array([iterations_i/2,
                #                       iterations_j/2,
                #                       iterations_k/2])
                # current_point = np.array([i_no, j_no, k_no])
                # euclidean_distance = np.linalg.norm(current_point - mid_point)

                # Approximate standardisation
                # euclidean_distance = 1 - (euclidean_distance / 140)
            else:
                euclidean_distance = 0
                lin_grad_patch = np.zeros([self.patch_size,
                                           self.patch_size,
                                           self.patch_size,
                                           1])

            secondary_input = lin_grad_patch

            # spatial_patch = np.full([self.patch_size,
            #                          self.patch_size,
            #                          self.patch_size], euclidean_distance)

            # euclidean_distance = [spatial_patch]

        # Decide whether a new volume has to be loaded next time round
        if self.total_indices - 1 == unshuffled_index:
            # print("total_indices is %s and unshuffled_index is %s" % (total_indices, unshuffled_index))
            if only_centre:
                self.y_offset += self.total_indices
                self.y_volume_needed = True

                self.next_y += 1
            else:
                # print("\nx_offset before changing the offset by total_indices is ", self.x_offset)
                self.x_offset += self.total_indices
                # print("x_offset after changing the offset by total_indices is ", self.x_offset)
                self.x_volume_needed = True

                self.next_x += 1

        if only_centre:
            return patch
        else:
            return patch, secondary_input

    def create_shuffling_dict(self, seed):

        random.seed(seed)
        # Create a mapping between requested and retrieved index, if shuffling
        requested_indices = list(range(self.total_indices))
        shuffled_indices = random.sample(requested_indices, self.total_indices)
        mapping_dict = {i: shuffled_indices[i] for i in requested_indices}
        reverse_mapping_dict = {shuffled_indices[i]: i
                                for i in requested_indices}

        self.mapping_dict = mapping_dict
        self.reverse_mapping_dict = reverse_mapping_dict
        return

    def patch_from_volume(self, index):
        """Only supports u-nets at the moment. Given an index, it will find
        and return the corresponding single patch for x and y"""

        if self.x_volume_needed:
            if self.randomise_spatial_offset:
                self.spatial_offset = np.random.randint(0, self.patch_size // 2)
                #print("Spatial offset set to ", self.spatial_offset)

            # If only one patient is given, it only has to loop over self.x_lists directlyt
            if self.one_patient_only:
                self.x_volumes = [
                    self.load_volume(path) for path in self.x_lists]
            else:
                self.x_volumes = [
                    self.load_volume(path) for path in self.x_lists[self.next_x]]

            self.x_volumes = [self.scale(volume) for volume in self.x_volumes]

            # Get linear gradients if secondary input require
            if self.secondary_input:
                self.linear_gradients = self.get_linear_gradients(
                    self.x_volumes[0].shape)

            self.x_volume_needed = False

            if self.shuffle: self.create_shuffling_dict(np.random.seed())

            if self.one_patient_only:
                self.y_volumes = [
                    self.load_volume(path) for path in self.y_lists]
            else:
                self.y_volumes = [
                    self.load_volume(path) for path in self.y_lists[self.next_y]]

            self.y_volume_needed = False

            # Process segmentations with one channel and multiple values in it
            if self.BraTS_like_y:
                volume = self.y_volumes[0]
                self.y_volumes = []
                class_values = [0, 1, 2, 4]
                for i in class_values:
                    # Skip the air class, which is added later anyway
                    if i == 0:
                        continue
                    self.y_volumes.append(volume == i)

        # First, apply the offsets to the index and then shuffle it
        unshuffled_index = index - self.y_offset
        if self.shuffle:
            shuffled_index = self.mapping_dict[unshuffled_index]
        else:
            shuffled_index = unshuffled_index

        # necessary?
        offset = self.y_offset
        vol_dims = self.y_volumes[0].shape

        y_volumes = self.y_volumes

        # necessary?
        offset = self.x_offset
        vol_dims = self.x_volumes[0].shape

        x_volumes = self.x_volumes

        # Figuring out the corresponding patch for the requested index
        ij_area = self.iterations_i * self.iterations_j
        in_plane_index = shuffled_index % ij_area

        # This is the number of steps to take in each direction
        i_no = in_plane_index % self.iterations_i
        j_no = in_plane_index // self.iterations_i
        k_no = shuffled_index // ij_area

        # Convert from which position (i.e. 5th patch) to absolute location
        i = i_no * self.stride
        j = j_no * self.stride
        k = k_no * self.stride

        patcher = self.Patcher([i, j, k], self.patch_size)

        # Extract the y (label) information
        y_patches = [patcher.patch(volume) for volume in y_volumes]
        y_patches = np.moveaxis(y_patches, 0, 3)

        # Background (air) class
        no_brain = np.ones_like(y_patches[:, :, :, 0]) - np.sum(y_patches, axis=3)
        no_brain = np.expand_dims(no_brain, -1)

        y_patches = np.concatenate([y_patches, no_brain], axis=3)

        # Extract the x (image) information
        x_patches = [patcher.patch(volume) for volume in x_volumes]

        if self.secondary_input:
            secondary_input = patcher.patch(self.linear_gradients)
        else:
            secondary_input = np.zeros([self.patch_size,
                                        self.patch_size,
                                        self.patch_size,
                                        3])

        # Decide whether a new volume has to be loaded next time round
        if self.total_indices - 1 == unshuffled_index:
            # These can become one offset once this is tested and working
            self.y_offset += self.total_indices
            self.x_offset += self.total_indices

            self.y_volume_needed = True
            self.x_volume_needed = True

            self.next_y += 1
            self.next_x += 1

        return x_patches, y_patches, secondary_input

    def __load__(self, index):
        """Returns a single patch as requested by __getitem__. Could probably
        be combined with patch_from_volume, as this currently just calls that.
        However, patch_from_volume is currently unwieldy so that would have
        to be modularised first"""

        x_patches, y_patches, secondary_input = self.patch_from_volume(index)

        # Can try np.expand_dims( ) here instead
        x_patches = [patch_x.reshape(
            self.patch_size, self.patch_size, self.patch_size, 1)
            for patch_x in x_patches]

        # Treats multimodal images as separate channels
        x_patches = np.concatenate(x_patches, axis=3)

        # Combine multimodal / single modality x with the spatial inputs
        x_patches = np.concatenate([x_patches, secondary_input], axis=3)

        return x_patches, y_patches

    def on_epoch_end(self):

        #new_offset = np.random.randint(0, self.patch_size // 2)
        #print("\n" + "#" * 7 + "Epoch ended. Going to randomly change "
        #      "spatial offset to ", new_offset)
        #self.spatial_offset = new_offset

        #print("Going to shuffle the two lists")
        if not self.one_patient_only:
            combined = list(zip(self.x_lists, self.y_lists))
            random.shuffle(combined)
            self.x_lists[:], self.y_lists[:] = zip(*combined)

        self.x_offset = 0
        self.y_offset = 0

        self.next_x = 0
        self.next_y = 0

    def __getitem__(self, batch):

        # print(\n + "#" * 8 + "Requested batch %s from generator" % batch)
        # print("Generator has length ", len(self))

        # print("self.total_patches is %s, and highest index to request is %s"
        #       % (self.total_patches, (batch + 1)*self.batch_size))

        if self.prediction and batch != 0:
            if (batch + 1) * self.batch_size > self.total_patches:
                self.batch_size = self.total_patches - batch * self.batch_size
                # print("Got to the end of the volume - next batch will be smaller")
                # print("Next batch will have size ", self.batch_size)

        batch = [self.__load__(index) for index in
                 range((batch * self.batch_size), (batch + 1) * self.batch_size)]

        batch_x = [data_point[0] for data_point in batch]
        batch_y = [data_point[1] for data_point in batch]

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        x_1 = np.concatenate(batch_x[:, :, :, :, :self.no_of_xs], axis=0)
        x_1 = x_1.reshape(self.batch_size,
                          self.patch_size,
                          self.patch_size,
                          self.patch_size,
                          self.no_of_xs)

        if self.secondary_input:
            x_2 = np.concatenate(batch_x[:, :, :, :, self.no_of_xs:], axis=0)
            x_2 = x_2.reshape(self.batch_size,
                              self.patch_size,
                              self.patch_size,
                              self.patch_size,
                              3)

        if self.x_only:
            if self.secondary_input:
                return [x_1, x_2]
            else:
                return x_1
        else:
            if self.secondary_input:
                return [x_1, x_2], batch_y
            else:
                return x_1, batch_y

