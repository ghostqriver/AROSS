

from pyclustering.container.kdtree import kdtree,node,kdtree_balanced
from pyclustering.utils import find_left_element
import operator
# from COS_Funcs.dist import calc_cov_i,calc_dist
from dist import calc_cov_i,calc_dist
COMPARE_CHILD = {
    0: (operator.le, operator.sub),
    1: (operator.ge, operator.add),
}
class kdtree_node(node):
    
    def children(self):
        """
        Returns an iterator for the children of the Node.
        The children are returnd as (Node, pos) tuples where pos is 0 for the
        left subnode and 1 for the right subnode.
        """
        if self.left and self.left.data is not None:
            yield self.left, 0
        if self.right and self.right.data is not None:
            yield self.right, 1
            
    def _search_node(self, point, k, results, examined, L=2,cov_i=None):

        examined.add(self)

        # Get current best
        if len(results)==0:
            # results is empty
            bestNode = None
            bestDist = float('inf')
        else:
            # find the nearest (node, distance) tuple
            bestNode, bestDist = sorted(
                results.items(), key=lambda n_d: n_d[1], reverse=True)[0]

        nodesChanged = False

        # If the current node is closer than the current best, then it becomes
        # the current best. And the maximum distance nodes should be removed.
        nodeDist = calc_dist(self.data,point,L,cov_i=cov_i)
        
        if nodeDist < bestDist:
            if len(results) == k and bestNode:
                # if full, remove one
                # results.pop(bestNode)
                # here is the difference, i remove the max dist node
                maxNode, maxDist = sorted(
                    results.items(), key=lambda n: n[1], reverse=True)[0]
                results.pop(maxNode)

            results[self] = nodeDist
            nodesChanged = True
        
        # If we're equal to the current best, add it, regardless of k
        elif nodeDist == bestDist:
            results[self] = nodeDist
            nodesChanged = True
        
        # If we don't have k results yet, add it anyway
        elif len(results) < k:
            results[self] = nodeDist
            nodesChanged = True

        # Get new best only if nodes have changed
        # !!!!!!!! why best is the minimum one, it should be the maximum one.. otherwise the best is the best,
        # But other neighbors will be blocked by this best
        if nodesChanged:
            # The minimum distance one as the best
            bestNode, bestDist = sorted(
                results.items(), key=lambda n: n[1], reverse=True)[0]

        # Check whether there could be any other points on the other side
        # of the splitting.
        # hyperplane that are closer to the search point than the current best.
        for child, pos in self.children():
            if child in examined:
                continue

            examined.add(child)
            compare, combine = COMPARE_CHILD[pos]

            # Since the hyperplanes are all axis-aligned this is implemented
            # as a simple comparison to see whether the difference between the
            # splitting coordinate of the search point and current node is less
            # than the distance (overall coordinates) from the search point to
            # the current best.
            nodePoint = self.data[self.disc]
            pointPlusDist = combine(point[self.disc], bestDist) # right child : add    left child : sub
            lineIntersects = compare(pointPlusDist, nodePoint) # right child : gt     left child: lt

            # If the hypersphere crosses the plane, there could be nearer
            # points on the other side of the plane, so the algorithm must move
            # down the other branch of the tree from the current node looking
            # for closer points, following the same recursive process as the
            # entire search.
            if lineIntersects:
#             if True:
                child._search_node(point, k, results, examined, L,cov_i=cov_i)
    
    
class kdtree_(kdtree):  
    
    def __init__(self, points, payloads=None, cov_i=None):
        """!
        @brief Initializes balanced static KD-tree.
        @param[in] points (array_like): Points that should be used to build KD-tree.
        @param[in] payloads (array_like): Payload of each point in `points`.
        """
        if points is None:
            self._length = 0
            self._dimension = 0
            self._point_comparator = None
            self._root = None
            return

        self._dimension = len(points[0])
        self._point_comparator = self._create_point_comparator(type(points[0]))
        self._length = 0

        nodes = []
        for i in range(len(points)):
            payload = None
            if payloads is not None:
                payload = payloads[i]

            nodes.append(kdtree_node(points[i], payload, None, None, -1, None))

        self._root = self.__create_tree(nodes, None, 0)
        if cov_i is None:
            self.cov_i = calc_cov_i(points)
    
    def __create_tree(self, nodes, parent, depth):
        """!
        @brief Creates balanced sub-tree using elements from list `nodes`.
        @param[in] nodes (list): List of KD-tree nodes.
        @param[in] parent (node): Parent node that is used as a root to build the sub-tree.
        @param[in] depth (uint): Depth of the tree that where children of the `parent` should be placed.
        @return (node) Returns a node that is a root of the built sub-tree.
        """
        if len(nodes) == 0:
            return None

        discriminator = depth % self._dimension

        nodes.sort(key=lambda n: n.data[discriminator])
        median = len(nodes) // 2

        # Elements could be the same around the median, but all elements that are >= to the current should
        # be at the right side.
        # TODO: optimize by binary search - no need to use O(n)
        median = find_left_element(nodes, median, lambda n1, n2: n1.data[discriminator] < n2.data[discriminator])
        # while median - 1 >= 0 and \
        #         nodes[median].data[discriminator] == nodes[median - 1].data[discriminator]:
        #     median -= 1

        new_node = nodes[median]
        new_node.disc = discriminator
        new_node.parent = parent
        new_node.left = self.__create_tree(nodes[:median], new_node, depth + 1)
        new_node.right = self.__create_tree(nodes[median + 1:], new_node, depth + 1)

        self._length += 1
        return new_node

    def insert(self, point, payload=None):
        """!
        @brief Insert new point with payload to kd-tree.
        
        @param[in] point (list): Coordinates of the point of inserted node.
        @param[in] payload (any-type): Payload of inserted node. It can be ID of the node or
                    some useful payload that belongs to the point.
        
        @return (node) Inserted node to the kd-tree.
        
        """
        
        if self._root is None:
            self._dimension = len(point)
            self._root = kdtree_node(point, payload, None, None, 0)
            self._point_comparator = self._create_point_comparator(type(point))

            self._length += 1
            return self._root

        cur_node = self._root

        while True:
            discriminator = (cur_node.disc + 1) % self._dimension

            if cur_node.data[cur_node.disc] <= point[cur_node.disc]:
                if cur_node.right is None:
                    cur_node.right = kdtree_node(point, payload, None, None, discriminator, cur_node)

                    self._length += 1
                    return cur_node.right
                else: 
                    cur_node = cur_node.right
            
            else:
                if cur_node.left is None:
                    cur_node.left = kdtree_node(point, payload, None, None, discriminator, cur_node)

                    self._length += 1
                    return cur_node.left
                else:
                    cur_node = cur_node.left
        
    def search_knn(self, point, k, L=2,cov_i=None):
        
        if cov_i is None:        
            cov_i = self.cov_i
        prev = None
        cur_node = self._root

        # go down the trees as we would for inserting
        while cur_node is not None:
            if  point[cur_node.disc] < cur_node.data[cur_node.disc]:
                # go to left subtree
                prev = cur_node
                cur_node = cur_node.left
            else:
                # go to right subtree
                prev = cur_node
                cur_node = cur_node.right

        if prev is None:
            return []

        examined = set()
        results = {}

        # Go up the tree, looking for better solutions
        cur_node = prev
        while cur_node is not None:
            cur_node._search_node(point, k, results, examined, L=L, cov_i=cov_i)
            cur_node = cur_node.parent

        return sorted(results.items(), key=lambda a: a[1])