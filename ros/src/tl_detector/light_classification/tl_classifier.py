from styx_msgs.msg import TrafficLight
import rospy
import tensorflow as tf
import numpy as np
import datetime

THRESHOLD = 0.5

CLASSES = ['green', 'red', 'yellow', 'unkonwn']

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        ## load trained TF model
        model_path = r'light_classification/model.pb'

        self.graph = tf.Graph()

        with self.graph.as_default():
            graph_df = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as g:
                graph_df.ParseFromString(g.read())
                tf.import_graph_def(graph_df, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.graph)    

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        with self.graph.as_default():
            img_input = np.expand_dims(image, axis=0)
            # t = datetime.datetime.now()
            (boxes, scores, classes, _) = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections], 
                    feed_dict={self.image_tensor: img_input})
            # elapse = datetime.datetime.now() - t
            # rospy.logwarn("One inference with: {0}s".format(elapse))

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # rospy.logwarn("boxes: {0}s".format(boxes))
        # rospy.logwarn("scores: {0}s".format(scores))
        # rospy.logwarn("classes: {0}s".format(classes))

        score = scores[0]
        cl = classes[0]

        rospy.loginfo('Score: {0}  and Class: {1}'.format(score, CLASSES[cl-1]))
        
        if score > THRESHOLD:
            if cl == 1:
                rospy.loginfo('Green')
                return TrafficLight.GREEN
            elif cl == 2:
                rospy.loginfo('Red')
                return TrafficLight.RED
            elif cl == 3:
                rospy.loginfo('YELLOW')
                return TrafficLight.YELLOW

        rospy.loginfo('UNKOWN')
        return TrafficLight.UNKNOWN
