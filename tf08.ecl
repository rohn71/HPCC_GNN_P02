IMPORT PYTHON3 AS PYTHON;
IMPORT Std;
IMPORT GNN.Tensor;
IMPORT GNN.Tensor AS GT;
IMPORT $.Types;
IMPORT Std.System.thorlib;

node:= thorlib.node();
nNodes:= thorlib.nodes();


t_Tensor := Tensor.R4.t_Tensor;
tensdata := Tensor.R4.tensData;


EXPORT tf08 := MODULE


    EXPORT STREAMED DATASET(tensdata) pyConvertImages(STREAMED DATASET(Types.ImgRec) imgs, INTEGER theight, INTEGER twidth, INTEGER tchannel, INTEGER tmode) := EMBED(Python:activity)
        import numpy
        import matplotlib.pyplot as plt
        import io
        import tensorflow as tf
        from tensorflow import keras
        import math
        from tensorflow.python.ops.numpy_ops import np_config 
        np_config.enable_numpy_behavior()
        
        global Np2Tens


        # Returns a streamed dataset of t_Tensor
        image_tensors = []


        def resize_image(image, target_size):
            image = tf.image.resize(image, target_size,preserve_aspect_ratio=False)
            return image

        def crop_fill(image, target_size):
            image_shape = tf.shape(image)[:2]
            target_height, target_width, target_channel = target_size

            # Calculate the scaling factor for resizing
            scale_factor = tf.cast(target_height / image_shape[0], tf.float32)

            # Resize the image maintaining aspect ratio
            new_width = tf.cast(scale_factor * tf.cast(image_shape[1], tf.float32), tf.int32)
            image = resize_image(image, (target_height, new_width))

            # Pad or crop the image to match the desired size
            image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)

            return image


        def fit(image, target_size):
            image_shape = tf.shape(image)
            target_height, target_width , target_channel = target_size

            # Calculate the scaling factors for resizing
            scale_factor_h = tf.cast(target_height / image_shape[0], tf.float32)
            scale_factor_w = tf.cast(target_width / image_shape[1], tf.float32)
            #image_channels = tf.cast(image_shape[-1], tf.int32)
            scale_factor = tf.minimum(scale_factor_h, scale_factor_w)
            # Resize the image maintaining aspect ratio
            new_height = tf.cast(scale_factor * tf.cast(image_shape[0], tf.float32), tf.int32)
            new_width = tf.cast(scale_factor * tf.cast(image_shape[1], tf.float32), tf.int32)
            #new_channel = tf.cast(adjusted_channel[-1], tf.int32)
            image = resize_image(image, (new_height, new_width))
            # Pad or crop the image to match the desired size
            image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)

            return image


        def fitv(image, target_size):
            image_shape = tf.shape(image)[:2]
            target_height, target_width, target_channel = target_size

            # Calculate the scaling factor for resizing
            scale_factor = tf.cast(target_height / image_shape[0], tf.float32)

            # Resize the image maintaining aspect ratio
            new_width = tf.cast(scale_factor * tf.cast(image_shape[1], tf.float32), tf.int32)
            image = resize_image(image, (target_height, new_width))

            # Pad or crop the image to match the desired size
            image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)

            return image

        def fith(image, target_size):
            image_shape = tf.shape(image)[:2]
            target_height, target_width, target_channel = target_size

            # Calculate the scaling factor for resizing
            scale_factor = tf.cast(target_width / image_shape[1], tf.float32)

            # Resize the image maintaining aspect ratio
            new_height = tf.cast(scale_factor * tf.cast(image_shape[0], tf.float32), tf.int32)
            image = resize_image(image, (new_height, target_width))

            # Pad or crop the image to match the desired size
            image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)

            return image


        def adjust_channels(image, target_channels, preserve_alpha=True):
            num_channels = image.shape[-1]

            if num_channels == 1 and target_channels == 3:
                image = tf.tile(image, [1, 1, 3])
            elif num_channels == 3 and target_channels == 1:
                image = tf.reduce_mean(image, axis=-1, keepdims=True)
            elif num_channels == 3 and target_channels == 4 and preserve_alpha:
                alpha = tf.ones_like(image[..., :1])
                image = tf.concat([image, alpha], axis=-1)
            elif num_channels == 4 and target_channels == 3 and preserve_alpha:
                image = image[..., :3]
            elif num_channels == 4 and target_channels == 3 and not preserve_alpha:
                image = image[..., :3]
            elif num_channels == 4 and target_channels == 1:
                if preserve_alpha:
                    image = tf.reduce_mean(image[..., :3], axis=-1, keepdims=True)
                else:
                    image = tf.reduce_mean(image, axis=-1, keepdims=True)
            else:
                print("Invalid channel adjustment!")
                return None

            return image



        def process_image(image_array, method, target_size):
            image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

            switcher = {
                1: crop_fill,
                2: fit,
                3: fitv,
                4: fith
            }

            func = switcher.get(method, None)
            if func:
                image_tensor = func(image_tensor, target_size)
            else:
                print("Invalid method!")

            return image_tensor

        def _Np2Tens(a, wi=0, maxSliceOverride=0, isWeights = False):
            #   dTypeDict is used to convey the data type of a tensor.  It must be
            #   kept in sync with the Tensor data types in Tensor.ecl
            dTypeDict = {1:numpy.float32, 2:numpy.float64, 3:numpy.int32, 4:numpy.int64}
            dTypeDictR = {'float32':1, 'float64':2, 'int32':3, 'int64':4}
            #   Store the element size for each tensor data type.
            dTypeSizeDict = {1:4, 2:8, 3:4, 4:8}
            maxSliceLen = 1000000
            nNodes = 1
            nodeId = 0
            epsilon = .000000001
            origShape = list(a.shape)
            flatA = a.reshape(-1)
            flatSize = flatA.shape[0]
            currSlice = 1
            indx = 0
            #datType = dTypeDictR[str(a.dtype)]
            #elemSize = dTypeSizeDict[datType]

            datType = 1
            elemSize = 4
            if maxSliceOverride:
                maxSliceSize = maxSliceOverride
            else:
                maxSliceSize = divmod(maxSliceLen, elemSize)[0]
            if isWeights and nNodes > 1 and flatSize > nNodes:
                # When we are synchronizing weights, we need to make sure
                # that we create Tensor with at least 1 slice per node.
                # This allows all nodes to participate equally in the
                # aggregation of weight changes.  For other data, it
                # is more efficient to return fewer slices.
                altSliceSize = math.ceil(flatSize / nNodes)
                maxSliceSize = min([maxSliceSize, altSliceSize])
            while indx < flatSize:
                remaining = flatSize - indx
                if remaining >= maxSliceSize:
                    sliceSize = maxSliceSize
                else:
                    sliceSize = remaining
                dat = list(flatA[indx:indx + sliceSize])
                dat = [float(d) for d in dat]
                elemCount = 0
                for i in range(len(dat)):
                    if abs(dat[i]) > epsilon:
                        elemCount += 1
                if elemCount > 0 or currSlice == 1:
                    if elemCount * (elemSize + 4) < len(dat):
                        # Sparse encoding
                        sparse = []
                        for i in range(len(dat)):
                            if abs(dat[i]) > epsilon:
                                sparse.append((i, dat[i]))
                        yield (nodeId, wi, currSlice, origShape, datType, maxSliceSize, sliceSize, [], sparse)
                    else:
                        # Dense encoding
                        yield (nodeId, wi, currSlice, origShape, datType, maxSliceSize, sliceSize, dat, [])
                currSlice += 1
                indx += sliceSize

        #def selectMethod()

        def makeStack(ten):
            image_tensors.append(ten)
            tensor_stack = tf.stack(image_tensors)
            return tensor_stack
        
        Np2Tens = _Np2Tens

        def generateTensors(imageRecs):
            #method = "fit"  # Replace with your desired method
            target_size = (theight, twidth, tchannel)
            offset = -25  # Replace with your desired offset
            imgs = 0
            all_tens = numpy.array([1])
            recnum = 1
            for rec in imageRecs:
                row = column = channel = 1
                id,filename,img = rec
                tokens = filename.split('.')
                ext = tokens[1].lower()
                imgs += 1
                image_np = numpy.frombuffer(img, dtype='uint8')
                image = plt.imread(io.BytesIO(image_np), ext)
                image_tensor = process_image(image, tmode, target_size)
                adjusted_tensor = adjust_channels(image_tensor, tchannel, preserve_alpha=True)
                flat_tensor = tf.reshape(adjusted_tensor, [-1])
                for val in flat_tensor:
                    indexes = [recnum,row,column,channel]
                    yield (indexes,float(val))
                    if channel < tchannel:
                        channel += 1
                    elif column < twidth:
                        channel = 1
                        column += 1
                    else:
                        channel = 1
                        column = 1
                        row += 1
                recnum += 1


        try:
            return generateTensors(imgs)
            
        except:
            import traceback as tb
            exc = tb.format_exc()
            assert False, exc
    ENDEMBED;

    EXPORT DATASET(t_Tensor) convertImages(DATASET(Types.ImgRec) images, INTEGER targetheight, INTEGER targetwidth, INTEGER targetchannel, INTEGER transform_mode) := FUNCTION
        recspernode0 := COUNT(images)/nNodes;
        recspernode := IF(recspernode0 = TRUNCATE(recspernode0), recspernode0, TRUNCATE(recspernode0 + 1));
        imagesD := DISTRIBUTE(images,(id-1) DIV recspernode);
        td0 := pyConvertImages(imagesD,targetheight,targetwidth,targetchannel,transform_mode);
        td := PROJECT(td0,TRANSFORM(RECORDOF(LEFT),SELF.indexes:= [node*recspernode+LEFT.indexes[1],LEFT.indexes[2],LEFT.indexes[3],LEFT.indexes[4]], SELF:=LEFT));
        td_s := SORT(td, indexes);
        //shape := [0, target_height, target_channels];
        tensor := Tensor.R4.MakeTensor( [0,targetheight,targetwidth,targetchannel] ,td_s, wi := 1);
        return tensor;

    END;

    // EXPORT DATASET(t_Tensor) stackTensors(DATASET tensors_s) := FUNCTION
    //     tensors_s := SORT(tensors,wi,sliceId);
    //     return tensors_s;

    //EXPORT STREAMED DATASET(t_Tensor) pyConvertImages(STREAMED DATASET(Types.ImgRec) imgs) := EMBED(Python:activity)

END;
