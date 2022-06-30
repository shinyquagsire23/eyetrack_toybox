import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json 

from globals import *
from model import build_model

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph, _FunctionConverterDataInEager, _replace_variables_by_constants, _construct_concrete_function
from tensorflow.python.tools import freeze_graph
from tensorflow.core.framework import tensor_shape_pb2

from data_serve import data_serve_init, read_landmarks, read_iris_eli, data_serve_list
from resample import draw_patches

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

def export_network():
    info_dict = {}
    info_dict["box_info_tname"] = "UNDEFINED"
    info_dict["input_image_tname"] = "UNDEFINED"
    info_dict["boxes_tname"] = "UNDEFINED"
    info_dict["crop_1_tname"] = "UNDEFINED"
    info_dict["crop_2_tname"] = "UNDEFINED"
    info_dict["crop_3_tname"] = "UNDEFINED"
    info_dict["grid_x_1_tname"] = "UNDEFINED"
    info_dict["grid_x_2_tname"] = "UNDEFINED"
    info_dict["grid_x_3_tname"] = "UNDEFINED"
    info_dict["grid_y_1_tname"] = "UNDEFINED"
    info_dict["grid_y_2_tname"] = "UNDEFINED"
    info_dict["grid_y_3_tname"] = "UNDEFINED"
    info_dict["output_tname"] = "UNDEFINED"
    info_dict["manual_feedforward"] = []

    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    # TODO move this into its own function, also don't do it for every run bc it's slow

    
    print ("Build model...")
    model2 = build_model(1)
    print ("Compile model...")
    #model2.compile(loss="mean_squared_logarithmic_error", optimizer='adam', metrics=["accuracy"])
    print ("Load weights into model...")
    model2.load_weights('eyetrack_net.h5', by_name=True, skip_mismatch=True)
    #test_inputs = keras.Input(shape=Globals.EXPORT_INPUT_SHAPE)
    inputs = keras.Input(shape=Globals.INPUT_SHAPE_NOBATCH)
    print ("model test inputs...")
    #model2.call(inputs, training=False)
    print ("model.build...")
    #model2.build(Globals.EXPORT_INPUT_SHAPE)
    #model2.call(tf.zeros(Globals.EXPORT_INPUT_SHAPE), training=False)

    model2.summary()

    #inputs = tf.placeholder(tf.uint8, shape=Globals.EXPORT_INPUT_SHAPE)

    #inputs = keras.Input(shape=Globals.EXPORT_INPUT_SHAPE)

    print ("Save model...")
    #tf.keras.models.save_model(model=model2, filepath="eyetrack_net_saved", save_format="tf", )
    

    to_find_name_dict = {}

    model_basename = "eye_net/"
    info_dict["boxes_tname"] = model_basename + model2._fe_x_final.name
    #crop_tname = model2._crop_tensor.name
    to_find_name_dict["crop_1_tname"] = model2.hex_crop._patch_1.name.split(":0")[0]
    to_find_name_dict["crop_2_tname"] = model2.hex_crop._patch_2.name.split(":0")[0]
    to_find_name_dict["crop_3_tname"] = model2.hex_crop._patch_3.name.split(":0")[0]

    to_find_name_dict["grid_x_1_tname"] = model2.hex_crop._grid_x_1.name.split(":0")[0]
    to_find_name_dict["grid_x_2_tname"] = model2.hex_crop._grid_x_2.name.split(":0")[0]
    to_find_name_dict["grid_x_3_tname"] = model2.hex_crop._grid_x_3.name.split(":0")[0]

    to_find_name_dict["grid_y_1_tname"] = model2.hex_crop._grid_y_1.name.split(":0")[0]
    to_find_name_dict["grid_y_2_tname"] = model2.hex_crop._grid_y_2.name.split(":0")[0]
    to_find_name_dict["grid_y_3_tname"] = model2.hex_crop._grid_y_3.name.split(":0")[0]

    for n in to_find_name_dict:
        info_dict[n] = model_basename + to_find_name_dict[n] + ":0"#"UNDEFINED"

    print (to_find_name_dict)

    print ("Load model...")
    #imported = tf.saved_model.load("eyetrack_net_saved")
    print ("Loaded, procesisng...")

    #f = imported.signatures["serving_default"]
    print (model2.inputs)
    f =  tf.function(model2).get_concrete_function(tf.TensorSpec(Globals.EXPORT_INPUT_SHAPE, tf.uint8))
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

        for n in to_find_name_dict:
            if to_find_name_dict[n] == op.name.split("/")[-1]:
                info_dict[n] = op.name + ":0"
                print (n, op.name)
                break

        fresh_list = ["Placeholder"]
        #if op.type not in fresh_list:
        #    freeze_allowlist += [op.name]
        if (op.type == "AssignVariableOp" and "echo_rnn" in op.name):
            #print (op.inputs[0], op.inputs[0].shape)
            in_name = op.inputs[0].name.split(":0")[0]
            real_in_name = "UNDEFINED"
            fill_shape = None
            freeze_denylist += [in_name]
            for op_2 in frozen_func.graph.get_operations():
                if (op_2.name == in_name and len(op_2.inputs) != 0):
                    #print (op_2)
                    real_in_name = op_2.inputs[0].name.split(":0")[0]
                    keep_vars += [real_in_name]
                    break
                elif (op_2.name == in_name and len(op_2.inputs) == 0):
                    print (op_2)
                    real_in_name = op_2.name
                    keep_vars += [real_in_name]
                    break
            fill_shape = op.inputs[1].shape
            if fill_shape == (1,Globals.NETWORK_RECURRENT_SIZE) and "rnn" in op.inputs[1].name:
                box_info_tname = real_in_name + ":0"
            fill_shape = (1, fill_shape[1])
            entry = {}
            entry["tensor_to_fill"] = real_in_name + ":0"
            entry["tensor_to_read"] = op.inputs[1].name
            entry["identity_op_name"] = in_name + ":0"
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
        if (op.type == "Placeholder" and op.name not in keep_vars):
            print (op)
            freeze_allowlist += [op.name]
            a='a'
        elif (op.type == "ReadVariableOp"):
            print (op, op.inputs[0].name)
            freeze_allowlist += [op.name]
            #freeze_allowlist += [op.inputs[0].name.split(":0")[0]]
    #print (manual_feedforward)
    #print ("----------")
    #print (freeze_denylist)
    #print (freeze_allowlist)
    #print ("----------")

    #frozen_func, graph_def = my_convert_variables_to_constants_v2_as_graph(f, freeze_denylist=freeze_denylist)
    frozen_func, graph_def = my_convert_variables_to_constants_v2_as_graph(f, freeze_allowlist=freeze_allowlist)

    # We kinda have to search around for all of the intermediates,
    # because apparently tensor names mean literally nothing to TF2.x now...
    input_image_tname = "UNDEFINED"
    #boxes_tname = "UNDEFINED"
    #crop_1_tname = "UNDEFINED"
    output_tname = "UNDEFINED"

    
    for entry in manual_feedforward:
        tensor_to_fill = entry["tensor_to_fill"]
        identity_op_name = entry["identity_op_name"]
        shape_val = entry["shape"]

        #print (tensor_to_fill)
        identity_op_node = None
        for node in graph_def.node:
            #print (node.name)
            if node.name == tensor_to_fill.split(":")[0] and identity_op_name == tensor_to_fill:
                node.attr["dtype"].type = tf.float32.as_datatype_enum
                node.attr["shape"].shape.CopyFrom(tensor_shape_pb2.TensorShapeProto(dim=[
                  tensor_shape_pb2.TensorShapeProto.Dim(size=dim)
                  for dim in shape_val
              ]))
            elif node.name == tensor_to_fill.split(":")[0]:
                node.attr["dtype"].type = tf.float32.as_datatype_enum
                #print (node)
            elif node.name == identity_op_name.split(":")[0]:
                node.attr["T"].type = tf.float32.as_datatype_enum
                identity_op_node = node

            if identity_op_node is None and node.op == "Identity":
                print (node)
                identity_op_node = node

        for node in graph_def.node:
            if node.op == "ReadVariableOp":
                node.op = "Identity"
                node.attr["T"].type = tf.float32.as_datatype_enum
            elif node.op == "AssignVariableOp":
                print (node)
                real_out_name = node.input[1]
                keep_name = node.name
                keep_dtype = node.attr["dtype"].type
                node.CopyFrom(identity_op_node)
                node.name = keep_name
                node.input[0] = real_out_name
                node.attr["dtype"].type = keep_dtype
                #print (node)
    

    #manual_feedforward = []

    print("-" * 60)
    print("Frozen model layers: ")
    for op in frozen_func.graph.get_operations():
        #print(op.name, op.type)
        if (op.type == "CropAndResize"):
            #print (op.inputs)
            info_dict["crop_1_tname"] = op.name + ":0"
            info_dict["boxes_tname"] = op.inputs[1].name
        if (op.type == "RealDiv" and "per_image_standardization" in op.name):
            #info_dict["crop_1_tname"] = op.name + ":0"
            a='a'
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    input_image_tname = frozen_func.inputs[0].name
    output_tname = frozen_func.outputs[0].name

    #print (output_tname, crop_1_tname, boxes_tname)

    print ("Writing frozen graph")

    tf.io.write_graph(graph_or_graph_def=graph_def,
                        logdir=".",
                        name="eyetrack_net.pb",
                        as_text=False)

    info_dict["box_info_tname"] = box_info_tname
    info_dict["input_image_tname"] = input_image_tname
    '''
    info_dict["boxes_tname"] = boxes_tname
    info_dict["crop_1_tname"] = crop_1_tname
    info_dict["crop_2_tname"] = crop_2_tname
    info_dict["crop_3_tname"] = crop_3_tname
    '''

    if info_dict["crop_2_tname"] == "UNDEFINED":
        info_dict["crop_2_tname"] = info_dict["crop_1_tname"]
    if info_dict["crop_3_tname"] == "UNDEFINED":
        info_dict["crop_3_tname"] = info_dict["crop_3_tname"]

    if info_dict["grid_x_2_tname"] == "UNDEFINED":
        info_dict["grid_x_2_tname"] = info_dict["grid_x_1_tname"]
    if info_dict["grid_y_2_tname"] == "UNDEFINED":
        info_dict["grid_y_2_tname"] = info_dict["grid_y_1_tname"]
    if info_dict["grid_x_3_tname"] == "UNDEFINED":
        info_dict["grid_x_3_tname"] = info_dict["grid_x_1_tname"]
    if info_dict["grid_y_3_tname"] == "UNDEFINED":
        info_dict["grid_y_3_tname"] = info_dict["grid_y_1_tname"]

    info_dict["output_tname"] = output_tname
    info_dict["manual_feedforward"] = manual_feedforward

    with open("eyetrack_net.json", "w") as f:
        json.dump(info_dict, f)

def convert_img(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

eval_outputs_array = []
eval_outputs_idx_map = {}
def run_network():
    global eval_outputs_array, eval_outputs_idx_map

    export_network()

    info_dict = {}
    with open('eyetrack_net.json') as f:
        info_dict = json.load(f)

    box_info_tname = info_dict["box_info_tname"]
    input_image_tname = info_dict["input_image_tname"]
    boxes_tname = info_dict["boxes_tname"]
    crop_1_tname = info_dict["crop_1_tname"]
    crop_2_tname = info_dict["crop_2_tname"]
    crop_3_tname = info_dict["crop_3_tname"]

    grid_x_1_tname = info_dict["grid_x_1_tname"]
    grid_x_2_tname = info_dict["grid_x_2_tname"]
    grid_x_3_tname = info_dict["grid_x_3_tname"]

    grid_y_1_tname = info_dict["grid_y_1_tname"]
    grid_y_2_tname = info_dict["grid_y_2_tname"]
    grid_y_3_tname = info_dict["grid_y_3_tname"]

    output_tname = info_dict["output_tname"]
    manual_feedforward = info_dict["manual_feedforward"]

    #print (f)
    #print(f(input_1=tf.zeros((64,64,64,1))))

    def read_file(fpath):
        f = open(fpath, "rb")
        ret = f.read()
        f.close()
        return ret

    config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0})

    video_fpath_ = Globals.DIKABLIS_VIDEOS_ROOT + data_serve_list()[0]
    if Globals.TRAIN_ON_ANNOTATED:
        video_fpath = Globals.DIKABLIS_ANNOTATIONS_ROOT + data_serve_list()[0] + "iris_seg_2D.mp4"
    else:
        video_fpath = video_fpath_
    cap = cv2.VideoCapture(video_fpath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    cv2.namedWindow('feed')
    #cv2.namedWindow('crop')

    if frameCount > 10:
        frameCount = 10

    #rows = read_landmarks(video_fpath_.replace("VIDEOS", "ANNOTATIONS")+"iris_lm_2D.txt", img_width, img_height)
    rows = read_iris_eli(video_fpath_.replace("VIDEOS", "ANNOTATIONS")+"iris_eli.txt", img_width, img_height)


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

        test_rot = 0.0
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
                
                add_fetch(crop_1_tname)
                add_fetch(crop_2_tname)
                add_fetch(crop_3_tname)
                add_fetch(output_tname)
                add_fetch(boxes_tname)
                add_fetch(grid_x_1_tname)
                add_fetch(grid_y_1_tname)
                add_fetch(grid_x_2_tname)
                add_fetch(grid_y_2_tname)
                add_fetch(grid_x_3_tname)
                add_fetch(grid_y_3_tname)

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
                
                cropped_frame_1 = convert_img(get_fetch(outputs, crop_1_tname), 0, 255, np.uint8)#np.array(crop_pre, dtype=np.uint8)
                cropped_frame_1 = cropped_frame_1.reshape(cropped_frame_1.shape[1:])
                cropped_frame_2 = convert_img(get_fetch(outputs, crop_2_tname), 0, 255, np.uint8)#np.array(crop_pre, dtype=np.uint8)
                cropped_frame_2 = cropped_frame_2.reshape(cropped_frame_2.shape[1:])
                cropped_frame_3 = convert_img(get_fetch(outputs, crop_3_tname), 0, 255, np.uint8)#np.array(crop_pre, dtype=np.uint8)
                cropped_frame_3 = cropped_frame_3.reshape(cropped_frame_3.shape[1:])

                eye_boxes = get_fetch(outputs, boxes_tname)[0]

                grid_x_1 = get_fetch(outputs, grid_x_1_tname)[0]
                grid_y_1 = get_fetch(outputs, grid_y_1_tname)[0]
                grid_x_2 = get_fetch(outputs, grid_x_2_tname)[0]
                grid_y_2 = get_fetch(outputs, grid_y_2_tname)[0]
                grid_x_3 = get_fetch(outputs, grid_x_3_tname)[0]
                grid_y_3 = get_fetch(outputs, grid_y_3_tname)[0]

                box_center = recurrent_states[box_info_tname][0][0:2]

                #print (grid_x_1)
                #print (cropped_frame.shape)
                print (eye_boxes, output, recurrent_states[box_info_tname][0][0:2], recurrent_states[box_info_tname][0][3:5], recurrent_states[box_info_tname][0][6:10])
                #recurrent_states[box_info_tname][0][4] = -0.001
                #recurrent_states[box_info_tname][0][5] = -0.01

                #recurrent_states[box_info_tname][0][0] = 0.4
                #recurrent_states[box_info_tname][0][1] = 0.4

                test_rot += 0.1
                #recurrent_states[box_info_tname][0][6] = test_rot # shape
                #recurrent_states[box_info_tname][0][7] = test_rot

                #print (test_rot)

                
                #cv2.imshow('crop', cropped_frame)

                #print (buf.shape)
                #print (output)
                start_pt = (int(eye_boxes[0] * img_width), int(eye_boxes[1] * img_height))
                end_pt = (int(eye_boxes[2] * img_width), int(eye_boxes[3] * img_height))
                #print (buf.shape, start_pt, end_pt)
                cv2.rectangle(buf, start_pt, end_pt, (0,255,0), 1)

                draw_patches(buf, grid_x_1, grid_y_1, grid_x_2, grid_y_2, grid_x_3, grid_y_3)

                buf[img_height-(Globals.NETWORK_PATCH_SIZE*2):img_height-(Globals.NETWORK_PATCH_SIZE*1),0:Globals.NETWORK_PATCH_SIZE,:] = cropped_frame_1
                buf[img_height-(Globals.NETWORK_PATCH_SIZE*2):img_height-(Globals.NETWORK_PATCH_SIZE*1),(Globals.NETWORK_PATCH_SIZE+0):(Globals.NETWORK_PATCH_SIZE+Globals.NETWORK_PATCH_SIZE),:] = cropped_frame_2
                buf[img_height-Globals.NETWORK_PATCH_SIZE:img_height,(Globals.NETWORK_PATCH_SIZE*0+0):(Globals.NETWORK_PATCH_SIZE*0+Globals.NETWORK_PATCH_SIZE),:] = cropped_frame_3

                #cv2.rectangle(buf, (int(box_center[0] * img_width), int(box_center[1] * img_height)), (int(box_center[0] * img_width), int(box_center[1] * img_height)), (0,255,0), 3)

                #lms_x = rows[frame_num]["LM_X"]
                #lms_y = rows[frame_num]["LM_Y"]
                #lms_x = output[:Globals.NETWORK_OUTPUT_SIZE//2]
                #lms_y = output[Globals.NETWORK_OUTPUT_SIZE//2:]
                lms_x = [output[0:1]]
                lms_y = [output[1:2]]

                for i in range(0, len(lms_x)):
                    lm_x = int(lms_x[i] * img_width)
                    lm_y = int(lms_y[i] * img_height)

                    if lm_x < 0 or lm_x > img_width or lm_y < 0 or lm_y > img_height:
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
                    lm_x = int(lms_x[i] * img_width)
                    lm_y = int(lms_y[i] * img_height)

                    start_pt = (lm_x, lm_y)
                    end_pt = (lm_x, lm_y)

                    cv2.rectangle(buf, start_pt, end_pt, (0,0,255), 3)

                if newframe:
                    cv2.imshow('feed', buf)
                
                    if cv2.waitKey(16) & 0xFF ==  ord('q'):
                        done = True
                        break

            cap.release()