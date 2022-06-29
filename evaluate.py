import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from globals import *
from model import build_model

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph, _FunctionConverterDataInEager, _replace_variables_by_constants, _construct_concrete_function
from tensorflow.python.tools import freeze_graph

from data_serve import data_serve_init, read_landmarks, read_iris_eli, data_serve_list

# TODO I'm not sure if I'll actually need to prevent some Variables from getting baked in
def my_convert_variables_to_constants_v2_as_graph(func,
                                      lower_control_flow=True,
                                      aggressive_inlining=False,
                                      freeze_allowlist=None,
                                      freeze_denylist=None):

  converter_data = _FunctionConverterDataInEager(
      func=func,
      lower_control_flow=lower_control_flow,
      aggressive_inlining=aggressive_inlining,
      variable_names_allowlist=freeze_allowlist,
      variable_names_denylist=freeze_denylist)

  output_graph_def, converted_input_indices = _replace_variables_by_constants(
      converter_data=converter_data)

  frozen_func = _construct_concrete_function(func, output_graph_def,
                                      converted_input_indices)

  return frozen_func, output_graph_def

eval_outputs_array = []
eval_outputs_idx_map = {}
def run_network():
    global eval_outputs_array, eval_outputs_idx_map

    # TODO move this into its own function, also don't do it for every run bc it's slow

    old_batch_size = Globals.TRAIN_BATCH_SIZE
    Globals.TRAIN_BATCH_SIZE = 1
    model2 = build_model(1)
    model2.compile(loss="mean_squared_logarithmic_error", optimizer='adam', metrics=["accuracy"])
    model2.load_weights('eyetrack_net.h5', by_name=True, skip_mismatch=True)
    #test_inputs = keras.Input(shape=Globals.EXPORT_INPUT_SHAPE)
    test_inputs = keras.Input(shape=Globals.INPUT_SHAPE_NOBATCH)
    model2(test_inputs, training=False)
    model2.build(Globals.EXPORT_INPUT_SHAPE)

    

    test_inputs = keras.Input(shape=Globals.EXPORT_INPUT_SHAPE)

    tf.keras.models.save_model(model=model2, filepath="eyetrack_net_saved", save_format="tf", )
    Globals.TRAIN_BATCH_SIZE = old_batch_size

    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


    imported = tf.saved_model.load("eyetrack_net_saved")


    f = imported.signatures["serving_default"]
    #print (f)
    freeze_denylist = []
    frozen_func, graph_def = my_convert_variables_to_constants_v2_as_graph(f, freeze_allowlist=[])

    # This entire section is kinda dumb tbh
    # For whatever reason, convert_variables_to_constants_v2_as_graph really has trouble with
    # TF2's recurrent state. So we have to go through and specify which stuff is OK to convert
    # to constants, and what is not, and then convert all of the recurrent state to manual
    # feed-in inputs.
    manual_feedforward = []
    pass_2_list = []
    freeze_allowlist = []
    keep_vars = []
    box_info_tname = "UNDEFINED"
    for op in frozen_func.graph.get_operations():
        #print (op.name, op.type)
        #if "Assign" in op.name:
        #    print (op.type)
        fresh_list = ["Placeholder"]
        #if op.type not in fresh_list:
        #    freeze_allowlist += [op.name]
        if (op.type == "AssignVariableOp"):
            #print (op.inputs[0])
            in_name = op.inputs[0].name.split(":0")[0]
            real_in_name = None
            fill_shape = None
            freeze_denylist += [in_name]
            for op_2 in frozen_func.graph.get_operations():
                if (op_2.name == in_name):
                    #print (op_2)
                    real_in_name = op_2.inputs[0].name.split(":0")[0]
                    keep_vars += [real_in_name]
                    break
            fill_shape = op.inputs[1].shape
            if fill_shape == (None,Globals.NETWORK_RECURRENT_SIZE) and "rnn" in op.inputs[1].name:
                box_info_tname = real_in_name + ":0"
            fill_shape = (1, fill_shape[1])
            entry = {}
            entry["tensor_to_fill"] = real_in_name + ":0"
            entry["tensor_to_read"] = op.inputs[1].name
            entry["identity_op_name"] = in_name
            entry["shape"] = fill_shape
            manual_feedforward += [entry]
            #print (entry)
    #print ("-------")
    for op in frozen_func.graph.get_operations():
        #print (op.name, op.type)
        #if "Assign" in op.name:
        #    print (op.type)
        fresh_list = ["Placeholder"]
        #if op.type not in fresh_list:
        #    freeze_allowlist += [op.name]
        if (op.type == "Placeholder" and "unknown" in op.name and op.name not in keep_vars):
            freeze_allowlist += [op.name]
        elif (op.type == "ReadVariableOp"):
            freeze_allowlist += [op.name]
    #print ("----------")
    #print (freeze_denylist)
    #print (freeze_allowlist)
    #print ("----------")

    #frozen_func, graph_def = my_convert_variables_to_constants_v2_as_graph(f, freeze_denylist=freeze_denylist)
    frozen_func, graph_def = my_convert_variables_to_constants_v2_as_graph(f, freeze_allowlist=freeze_allowlist)

    # We kinda have to search around for all of the intermediates,
    # because apparently tensor names mean literally nothing to TF2.x now...
    input_image_tname = "UNDEFINED"
    boxes_tname = "UNDEFINED"
    crop_tname = "UNDEFINED"
    output_tname = "UNDEFINED"

    for entry in manual_feedforward:
        tensor_to_fill = entry["tensor_to_fill"]
        identity_op_name = entry["identity_op_name"]

        '''for op in frozen_func.graph.get_operations():
            if op.name == tensor_to_fill:
                print (op)'''
        identity_op_node = None
        for node in graph_def.node:
            if node.name == tensor_to_fill.split(":")[0]:
                node.attr["dtype"].type = tf.float32.as_datatype_enum
                #print (node)
            elif node.name == identity_op_name.split(":")[0]:
                node.attr["T"].type = tf.float32.as_datatype_enum
                identity_op_node = node
        for node in graph_def.node:
            if node.op == "ReadVariableOp":
                node.op = "Identity"
                node.attr["T"].type = tf.float32.as_datatype_enum
            elif node.op == "AssignVariableOp":
                real_out_name = node.input[1]
                keep_name = node.name
                node.CopyFrom(identity_op_node)
                node.name = keep_name
                node.input[0] = real_out_name
                #print (node)

    print("-" * 60)
    print("Frozen model layers: ")
    for op in frozen_func.graph.get_operations():
        #print(op.name, op.type)
        if (op.type == "CropAndResize"):
            #print (op.inputs)
            crop_tname = op.name + ":0"
            boxes_tname = op.inputs[1].name
        if (op.type == "RealDiv" and "per_image_standardization" in op.name):
            #crop_tname = op.name + ":0"
            a='a'
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    input_image_tname = frozen_func.inputs[0].name
    output_tname = frozen_func.outputs[0].name

    print (output_tname, crop_tname, boxes_tname)

    tf.io.write_graph(graph_or_graph_def=graph_def,
                        logdir=".",
                        name="eyetrack_net.pb",
                        as_text=False)

    #print (f)
    #print(f(input_1=tf.zeros((64,64,64,1))))

    def read_file(fpath):
        f = open(fpath, "rb")
        ret = f.read()
        f.close()
        return ret

    config = tf.compat.v1.ConfigProto()

    video_fpath_ = Globals.DIKABLIS_VIDEOS_ROOT + data_serve_list()[0]
    if Globals.TRAIN_ON_ANNOTATED:
        video_fpath = Globals.DIKABLIS_ANNOTATIONS_ROOT + data_serve_list()[0] + "iris_seg_2D.mp4"
    else:
        video_fpath = video_fpath_
    cap = cv2.VideoCapture(video_fpath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    cv2.namedWindow('feed')
    #cv2.namedWindow('crop')

    if frameCount > 10:
        frameCount = 10

    #rows = read_landmarks(video_fpath_.replace("VIDEOS", "ANNOTATIONS")+"iris_lm_2D.txt", frameWidth, frameHeight)
    rows = read_iris_eli(video_fpath_.replace("VIDEOS", "ANNOTATIONS")+"iris_eli.txt", frameWidth, frameHeight)


    fc = 0
    ret = True

    graph_file = "eyetrack_net.pb"

    with tf.compat.v1.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            return_elements=None,
            name= "",
            producer_op_list=None)

        session = tf.compat.v1.Session(graph=graph, config=config)

        recurrent_states = {}

        for entry in manual_feedforward:
            recurrent_states[entry["tensor_to_fill"]] = np.random.uniform(np.float32(-1.0), np.float32(1.0), entry["shape"])

        done = False
        while not done:
            cap = cv2.VideoCapture(video_fpath)
            frame_num = 0
            ret = True
            skipping = 0
            newframe = False
            while ret:
                if skipping == 0:
                    ret, buf = cap.read()
                    buf_orig = buf.copy()
                    frame_num += 1
                    newframe = True
                else:
                    buf = buf_orig.copy()
                    newframe = False
                skipping += 1
                if skipping >= 1:
                    skipping = 0
                if not ret:
                    break

                inputs_feed_dict = {}
                inputs_feed_dict[input_image_tname] = np.expand_dims(buf[:,:,0:1], 0)

                eval_outputs_array = []
                eval_outputs_idx_map = {}
                def add_fetch(name):
                    global eval_outputs_array, eval_outputs_idx_map
                    eval_outputs_idx_map[name] = len(eval_outputs_array)
                    eval_outputs_array += [name]

                def get_fetch(outputs, name):
                    return outputs[eval_outputs_idx_map[name]]            
                
                add_fetch(crop_tname)
                add_fetch(output_tname)
                add_fetch(boxes_tname)

                for entry in manual_feedforward:
                    add_fetch(entry["tensor_to_read"])

                for entry in recurrent_states:
                    inputs_feed_dict[entry] = recurrent_states[entry]

                outputs = session.run(eval_outputs_array, inputs_feed_dict)

                for entry in manual_feedforward:
                    recurrent_states[entry["tensor_to_fill"]] = get_fetch(outputs, entry["tensor_to_read"])

                #print (recurrent_states)
                
                #print (outputs)
                output = np.reshape(get_fetch(outputs, output_tname), (Globals.NETWORK_OUTPUT_SIZE,))
                cropped_frame = np.array(get_fetch(outputs, crop_tname), dtype=np.uint8)
                cropped_frame = cropped_frame.reshape(cropped_frame.shape[1:])

                eye_boxes = get_fetch(outputs, boxes_tname)[0]
                #print (cropped_frame.shape)
                print (eye_boxes, output, recurrent_states[box_info_tname][0][0:2], recurrent_states[box_info_tname][0][3:5])
                #recurrent_states[box_info_tname][0][4] = -0.001
                #recurrent_states[box_info_tname][0][5] = -0.01

                
                #cv2.imshow('crop', cropped_frame)

                #print (buf.shape)
                print (output)
                start_pt = (int(eye_boxes[1] * buf.shape[1]), int(eye_boxes[0] * buf.shape[0]))
                end_pt = (int(eye_boxes[3] * buf.shape[1]), int(eye_boxes[2] * buf.shape[0]))
                print (buf.shape, start_pt, end_pt)
                cv2.rectangle(buf, start_pt, end_pt, (0,255,0), 1)


                #lms_x = rows[frame_num]["LM_X"]
                #lms_y = rows[frame_num]["LM_Y"]
                #lms_x = output[:Globals.NETWORK_OUTPUT_SIZE//2]
                #lms_y = output[Globals.NETWORK_OUTPUT_SIZE//2:]
                lms_x = [output[0:1]]
                lms_y = [output[1:2]]

                for i in range(0, len(lms_x)):
                    lm_x = int(lms_x[i] * frameWidth)
                    lm_y = int(lms_y[i] * frameHeight)

                    if lm_x < 0 or lm_x > frameWidth or lm_y < 0 or lm_y > frameHeight:
                        continue
                    start_pt = (lm_x, lm_y)
                    end_pt = (lm_x, lm_y)
                    #print (start_pt, end_pt)

                    cv2.rectangle(buf, start_pt, end_pt, (255,0,0), 3)

                #lms_x = rows[frame_num]["LM_X"]
                #lms_y = rows[frame_num]["LM_Y"]
                lms_x = [rows[frame_num-1]["CENTER_X"]]
                lms_y = [rows[frame_num-1]["CENTER_Y"]]

                for i in range(0, len(lms_x)):
                    lm_x = int(lms_x[i] * frameWidth)
                    lm_y = int(lms_y[i] * frameHeight)

                    start_pt = (lm_x, lm_y)
                    end_pt = (lm_x, lm_y)

                    cv2.rectangle(buf, start_pt, end_pt, (0,0,255), 3)

                if newframe:
                    cv2.imshow('feed', buf)
                
                    if cv2.waitKey(16) & 0xFF ==  ord('q'):
                        done = True
                        break

            cap.release()