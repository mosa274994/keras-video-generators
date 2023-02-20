"""
TODO: description of file
"""

import cv2 as cv
import numpy as np
import tensorflow.keras.preprocessing.image as kimage

from .generator import VideoFrameGenerator


class SplitFrameGenerator(VideoFrameGenerator):
    """
    TODO

    from VideoFrameGenerator:

    - rescale: float fraction to rescale pixel data (commonly 1/255.)
    - nb_frames: int, number of frames to return for each sequence
    - classes: list of str, classes to infer
    - batch_size: int, batch size for each loop
    - use_frame_cache: bool, use frame cache (may take a lot of memory for \
        large dataset)
    - shape: tuple, target size of the frames
    - shuffle: bool, randomize files
    - transformation: ImageDataGenerator with transformations
    - split: float, factor to split files and validation
    - nb_channel: int, 1 or 3, to get grayscaled or RGB images
    - glob_pattern: string, directory path with '{classname}' inside that
        will be replaced by one of the class list
    """

    def __init__(self, *args, sequence_time: int = None, **kwargs):
        super().__init__(no_epoch_at_init=True, *args, **kwargs)
        # self.sequence_time = sequence_time
        super().__init__(nb_frames=nb_frames + 1, *args, **kwargs)
        self.sample_count = 0
        self.vid_info = []
        self.__frame_cache = {}
        self.__init_length()
        self.on_epoch_end()

    def __init_length(self):
        count = 0
        print("Checking files to find possible sequences, please wait...")
        for filename in self.files:
            cap = cv.VideoCapture(filename)
            fps = cap.get(cv.CAP_PROP_FPS)
            frame_count = self.count_frames(cap, filename)
            cap.release()

            if self.sequence_time is not None:
                seqtime = int(fps * self.sequence_time)
            else:
                seqtime = int(frame_count)

            stop_at = int(seqtime - self.nbframe)
            step = np.ceil(seqtime / self.nbframe).astype(np.int) - 1
            i = 0
            while stop_at > 0 and i <= frame_count - stop_at: # modified condition to ignore short video
                self.vid_info.append(
                    {
                        "id": count,
                        "name": filename,
                        "frame_count": int(frame_count),
                        "frames": np.arange(i, i + stop_at)[::step][: self.nbframe],
                        "fps": fps,
                    }
                )
                count += 1
                i += 1

        print(
            "For %d files, I found %d possible sequence samples"
            % (self.files_count, len(self.vid_info))
        )
        self.indexes = np.arange(len(self.vid_info))

    def on_epoch_end(self):
        # prepare transformation to avoid __getitem__ to reinitialize them
        if self.transformation is not None:
            self._random_trans = []
            for _ in range(len(self.vid_info)):
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.vid_info) / self.batch_size))

    def get_validation_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            sequence_time=self.sequence_time,
            nb_frames=self.nbframe,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            glob_pattern=self.glob_pattern,
            _validation_data=self.validation,
        )

    def get_test_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            sequence_time=self.sequence_time,
            nb_frames=self.nbframe,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            glob_pattern=self.glob_pattern,
            _test_data=self.test,
        )

    def _get_frames(
        self, video, nbframe, shape, fps, total_frames, force_no_headers=False
    ) -> Optional[Iterable]:
        cap = cv.VideoCapture(video)
        total_frames = self.count_frames(cap, video, force_no_headers)
        orig_total = total_frames

        if total_frames % 2 != 0:
            total_frames += 1
        
        # Experiment with the frames_step (skip_rate)
        frame_step = floor(fps / 5)
        # TODO: fix that, a tiny video can have a frame_step that is
        # under 1
        frame_step = max(1, frame_step)
        # frames = []
        frame_i = 0
        
        frame_num = 0
        
        while frame_num+nbframe < total_frames:
            video.set(cv.CAP_PROP_POS_FRAMES, frame_num)
            batch_idx = 0
            frames = []
            for _ in range(nbframe):
                grabbed, frame = vid.read()          
                if not grabbed:
                    break
                self.__add_and_convert_frame(
                    frame, frame_i, frames, orig_total, shape, frame_step
                )
            frame_num += nbframe
            yield np.array(frames)

        cap.release()

        # if not force_no_headers and len(frames) != nbframe:
        #     # There is a problem here
        #     # That means that frame count in header is wrong or broken,
        #     # so we need to force the full read of video to get the right
        #     # frame counter
        #     return self._get_frames(video, nbframe, shape, fps, total_frames, force_no_headers=True)

        if force_no_headers and len(frames) != nbframe:
            # and if we really couldn't find the real frame counter
            # so we return None. Sorry, nothing can be done...
            log.error(
                f"Frame count is not OK for video {video}, "
                f"{total_frames} total, {len(frames)} extracted"
            )
            return None
    
    def __getitem__(self, idx):
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        images = []

        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        transformation = None

        for i in indexes:
            # prepare a transformation if provided
            if self.transformation is not None:
                transformation = self._random_trans[i]

            vid = self.vid_info[i]
            video = vid.get("name")
            fps = vid.get("fps")
            # frame_count = vid.get("frame_count")
            classname = self._get_classname(video)

            # create a label array and set 1 to the right column
            label = np.zeros(len(classes))
            col = classes.index(classname)
            label[col] = 1.0

            video_id = vid["id"]
            if video_id not in self.__frame_cache:
                frames: Iterable = self._get_frames(video, nbframe, shape, fps)
            else:
                frames: Iterable = self.__frame_cache[video_id]

            # apply transformation
            if transformation is not None:
                frames = [
                    self.transformation.apply_transform(frame, transformation)
                    for frame in frames
                ]

            # add the sequence in batch
            images.append(frames)
            labels.append(label)

        return np.array(images), np.array(labels)
