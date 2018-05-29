"""
This module contains two tree classes which are optimized for access performance. Every
node has a `name` and a `props` attribute to save data. All nodes are addressable via
the path, which is a iterable with all name attributes
in order.

 - The root has the path `[]` and the child1 `['child1']` in the latter case.::

    + ROOT
    +- child1

 - The root can have a name but never has a parent. So checking if a node is the root is
 done with::

    node.parent is None

 - Properties are usually dictionaries. But you can also implement a class to store them,
 this can e.g. have the advantage of saving memory in combination with `__slots__`.
 Special care has to be taken, with :class:`IndexingNode` and the `on_update` event,
 since the change of properties should notify the node signal.

 - moving a node within a tree is done by setting the `parent` attribute::
    node.parent = new_parent

 - To get fast access to nodes by a specific attribute, use :class:`IndexingNode`.

 - The tree is iterable, it will iterate in randomly order over the whole subtree

 - The tree can be converted to a set, where the keys are the paths, it is possble to
 compare to trees quite fast using
   this method.

Custom Properties
-----------------

The dict-like functions `update(other, send_signals)` and items need to be implemented for
 costum Property classes.To ensure that the node can propergate the `on_update` event
 properly as well, make shure you call the signals::

    class MyFancyProperties(object):
        __slots__ = ['_id', 'node']

        def __init__(self, node):
            self.node = node

        @property
        def id(self):
            return self._id

        @id.setter
        def id(self, value):
            self.node.on_update.send(self.node, other={'_id': value})
            self._id = value

        def get(self, key):
            return getattr(self, key, None)

        def items(self):
            return [('_id', getattr(self, '_id', None))]

        def update(self, other, send_signal=True):
            if send_signal:
                self.node.on_update.send(self.node, other=other)

            if 'id' in other:
                self._id = other['id']

"""
from copy import copy
from itertools import chain
from threading import RLock
import logging
from collections import namedtuple

from blinker import Signal

__author__ = 'crosscloud GmbH'
__version__ = '1.0.10'

#: Special value for on_update events `new_value` kwarg, it is used after the property has
#:  been deleted
DELETE = object()
DEFAULT_VALUE = object()

logger = logging.getLogger(__name__)


class NotifyingDict(dict):
    """ This class holds the properties of a node, and also notifies it if any changes
    happens. """

    __slots__ = ['on_update']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_update = Signal()

    def __setitem__(self, key, item):
        # send on_update only if the value was changed
        if self.on_update and self.get(key, object()) != item:
            self.on_update.send(other={key: item})
        super().__setitem__(key, item)

    def __delitem__(self, key):
        if key in self:
            if self.on_update:
                self.on_update.send(other={key: DELETE})
        super().__delitem__(key)

    def pop(self, key, alt=DEFAULT_VALUE):
        """
        D.pop(key[,alt]) -> value, remove specified key and return the corresponding value.
        If key is not found, alt is returned if given, otherwise KeyError is raised
        """

        if key in self:
            value = self.get(key)
            self.__delitem__(key)
            return value

        if alt is not DEFAULT_VALUE:
            return alt

        raise KeyError

    def update(self, other=None, send_signal=True, destructive=False, **kwargs):
        """ Additionally emits a signal for every changed and new property

        :param other: the dictionary with which to update
        :param send_signal: This can turn off the emission for a signal
        :param destructive: Remove entries if they are not present in other
        """

        if other is None:
            other = {}

        update_dict = copy(other)
        different = self != other

        if destructive:
            keys_to_be_deleted = self.keys() - other.keys()
            update_dict.update({key: DELETE for key in keys_to_be_deleted})
            for key in keys_to_be_deleted:
                super().__delitem__(key)

        if send_signal and self.on_update and different and update_dict:
            self.on_update.send(other=update_dict)

        super().update(other, **kwargs)

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        data = state
        self.update(data)

    def __reduce__(self):
        return NotifyingDict, (), self.__getstate__()


class Node:
    """
    A node for the tree
    """
    # pylint: disable=too-many-instance-attributes

    # well, we have loads of instances, this improves memory performance
    __slots__ = ['_children', '_parent', '_name', '_path', 'lock', '_props', '_props_cls',
                 'on_moved', 'on_delete', 'on_create', 'on_update', 'on_renamed', '__weakref__']

    SIGNALS = ['on_create', 'on_delete', 'on_moved', 'on_renamed', 'on_update']

    def __init__(self, name, parent=None, props=None, props_cls=NotifyingDict):
        # pylint: too-many-arguments
        self._children = dict()
        self._parent = None
        self._name = None
        self._path = []

        if name:
            self.name = name

        if parent is not None:
            # this is not a root node

            #: :class:`blinker.Signal` is called with the node and `old_parent` and
            #: `new_parent` as kwargs
            self.on_moved = parent.on_moved

            #: :class:`blinker.Signal` is called with the node
            self.on_delete = parent.on_delete

            #: :class:`blinker.Signal` is called with the the new node
            self.on_create = parent.on_create

            #: :class:`blinker.Signal` is called if properties of a node are changed,
            #: this works thanks to the
            #: :class:`NotifyingDict` for dict properties or to your correctly implemented
            #:  costum property class
            #: it is called with the source node and with `key`, `old_value` and
            #: `new_value` as kwargs. `old_value`
            #: might :const:`CREATE`, which implies it is a new poperty and it might be
            #: :const:`DELETED`, which implies
            #: it was deleted. If a exception is thrown in the :const:`CREATE` occasion,
            #: it will abort the current operation.
            self.on_update = parent.on_update

            #: :class:`blinker.Signal` is called if the name of node was changed, args:
            #: `node`, `old_name` and
            #: `new_name`
            self.on_renamed = parent.on_renamed

            self.parent = parent
            self.lock = parent.lock
        else:
            # this is a root node

            self.on_moved = Signal()
            self.on_delete = Signal()
            self.on_create = Signal()
            self.on_update = Signal()
            self.on_renamed = Signal()

            #: this is a member which can be used to lock the whole tree, by a
            #:  :class:`threading.RLock`.
            self.lock = RLock()

        # the type which should be used for properties, by default class:`NotifyingDict`
        self._props_cls = props_cls
        self._props = self._props_cls()
        self._props.on_update.connect(self._on_dict_update)
        if props is not None:
            self.props.update(props, send_signal=False, destructive=True)

    def _on_dict_update(self, _, other):
        if self.on_update:
            self.on_update.send(self, other=other)

    @property
    def props(self):
        """Getter for the nodes props"""
        return self._props

    @props.setter
    def props(self, props):
        if props is not None:
            self._props.update(props, send_signal=True, destructive=True)
        else:
            self._props = self._props_cls(self)

    @property
    def children(self):
        """
        The children of this node

        :return: list of :class:`Node` objects
        """
        return self._children.values()

    @property
    def name(self):
        """
        The node's name. If set it will update the name in the parent as well. This can be
         a slow operation, if the node has a lot of children.

        :returns: string
        """
        return self._name

    @name.setter
    def name(self, name):
        """ name setter """
        if self._name == name:
            # if nothing changed -> return
            return

        # pylint: disable=W0212
        oldname = self._name
        self._name = name

        # rename in the parent dict, if there is a parent
        if self.parent is not None:
            del self.parent._children[oldname]
            self.parent._children[name] = self
        self._det_path()

        if oldname is not None and self.on_renamed:
            self.on_renamed.send(self, old_name=oldname)

    @property
    def parent(self):
        """
        The parent of the node. If set it updates the path of this node and of all
        children. That can be an expensive operation.
        """
        return self._parent

    @parent.setter
    def parent(self, parent):
        """ Setter for the parent. """
        # pylint: disable=protected-access

        if self._parent is parent:
            # if nothing changed -> return
            return

        # check if we are going to overwrite an other sibling
        if parent is not None and self.name in parent._children:
            conflicting_node = parent._children[self.name]
            raise DuplicateChildError(
                'A child with path "{}" with props "{}" already exists '
                'in the parent'.format(conflicting_node.path,
                                       conflicting_node.props))

        # for move operations
        old_parent = self._parent

        self._parent = parent

        if old_parent is not None:
            # Inform the previous parent to forget about this child.
            old_parent._children.pop(self.name)

        if parent is not None:
            self._parent._children[self.name] = self
        self._det_path()

        # it is only a move operation if it had a parent before
        if old_parent is not None and self.on_moved:
            self.on_moved.send(self, old_parent=old_parent)

    def _det_path(self):
        """  determines the path of this node and all its childsren """
        if self.parent is not None:
            self._path = copy(self.parent.path)
        else:
            self._path = []
        self._path.append(self.name)
        for child in self.children:
            # pylint: disable=W0212
            child._det_path()

    @property
    def path(self):
        """
        Returns the path traversed from the root dir
        :return: the path
        """
        return self._path

    @property
    def depth(self):
        """
        Calculates the depth in the tree
        :return:
        """
        return len(self.path)

    def subtree_equals(self, other):
        """
        Checks if another node is equal (inclusive subtree)


        :param other: The other tree
        :return: True or False
        """
        # pylint: disable=W0212
        if self != other:
            return False

        if self._children != other._children:
            return False

        return True

    def add_child(self, name, props=None):
        """
        Adds a child. This is the *only* way a child node should be created. It creates an
        instance of the same type it is.

        :param name: the child's name
        :param props: a dict of +props, passed to the ctor of the node
        :return: the new created Node
        """
        # creates a new node in the type of the instance
        new_node = type(self)(name=name, parent=self, props=props,
                              props_cls=type(self.props))
        try:
            if self.on_create:
                self.on_create.send(new_node)
            return new_node
        except:
            # rollback, if the handler raises
            self._children.pop(name, None)
            raise

    def has_child(self, name):
        '''
        Returns true if node has child with name
        '''
        return name in self._children

    def get_node(self, path):
        """
        Returns the child defined by the path
        :param path:
        :return:
        """
        # pylint: disable=W0212
        node = self
        for elem in path:
            node = node._children[elem]
        return node

    def get_node_safe(self, path):
        """ Like get_node, but implicitly creates the child nodes
        :param path:
        :return:
        """
        node = self
        for elem in path:
            if elem in node._children:  # pylint: disable=W0212
                node = node._children[elem]  # pylint: disable=W0212
            else:
                node = node.add_child(elem)
        return node

    def setdefault(self, path, props=None):
        """ like dict.setdefault, the node to the path does not exist, it will be
        created with the props, if it exists the props will be assigned to the existing
        node """
        node = self
        for elem in path:
            if elem in node._children:  # pylint: disable=W0212
                node = node._children[elem]  # pylint: disable=W0212
                if node.path == path and props:
                    # the node already exists, set the  props
                    node.props = props
            else:
                if node.path == path[:-1] and props:
                    # check if that is the parent of the future child, if yes create it
                    # with the props given to this function
                    # TODO: unittest dict thingy
                    node = node.add_child(elem, props=dict(props))
                else:
                    node = node.add_child(elem)
        return node

    def delete(self):
        """
        deletes the node from the tree
        :return: None
        """
        # pylint: disable=W0212

        if self.on_delete:
            self.on_delete.send(self)

        if self._parent:
            del self.parent._children[self.name]
        self._parent = None

        # remove all callbacks
        for node in self:
            node.on_create = None
            node.on_delete = None
            node.on_moved = None
            node.on_renamed = None
            node.on_update = None

    @property
    def iter_up(self):
        """ iterates over all parents starting with self """
        parent = self
        while parent.parent is not None:
            yield parent
            parent = parent.parent
        yield parent

    def iter_up_existing(self, path):
        """
        Iterates the existing nodes of a path
        if this is a path where some parts are not existing
        it iterates over the existing ones
        if this is called from a none-root node the iteration stops whit the node
        itself
        :param path: the path
        :return: iterator
        """

        # find the first existing node of the past
        node = None
        current_path = path
        while node is None:
            try:
                node = self.get_node(current_path)
            except KeyError:
                current_path = current_path[:-1]
                if current_path == []:
                    # at least return self
                    yield self
                    return

        # use the iter_up but stop with the current node
        # and do not go up until root
        for elem in node.iter_up:
            if elem is not self:
                yield elem
            else:
                yield self
                return

    # pylint: disable=no-self-use
    def create_instance(self, name=None):
        """ a factory function which creates an instance with the same type as
        this node, but not attached to it """
        # pylint: disable=no-self-use
        return Node(name=name)

    def __repr__(self):
        return '<Node at {} name="{}">'.format(id(self), getattr(self, 'name', '<unknown-name>'))

    def __iter__(self):
        """ Iterates over the node and all it's subnodes

        :return: the iterator
        """
        for child in chain(*self.children):
            yield child
        yield self

    def __len__(self):
        return sum(1 for _ in self)

    def __hash__(self):
        return hash(tuple(self.path))

    def __eq__(self, other):
        return self.path == other.path

    def __ne__(self, other):
        return not self == other

    def __getstate__(self):
        state = {'_children': self._children,
                 '_parent': self._parent,
                 '_name': self._name,
                 '_path': self._path,
                 '_props': self._props,
                 '_props_cls': self._props_cls}

        return state

    def __setstate__(self, state):
        """Responsible to be able to copy and pickle trees correctly.

        This is a bit complicated because all the signals and the lock function are created in the
        root and then are just referenced to the children.
        """
        #  (we are playing ctor)
        # pylint: disable=protected-access

        # recover all possible arguments from the state
        for attr, val in state.items():
            setattr(self, attr, val)

        if state['_parent'] is None:
            # This is the ROOT NODE (might be called as the first one, might not)

            # all signals and the lock are generated freshly here
            for sig in self.SIGNALS:
                setattr(self, sig, Signal())

            self._props.on_update.connect(self._on_dict_update)
            self.lock = RLock()

            # now it is assigned to all child nodes which are already reconstructed
            nodestack = list(self._children.values())
            while nodestack:
                node = nodestack.pop()
                if hasattr(node, '_children'):
                    node._set_child_node(self)
                    nodestack.extend(node.children)

        elif hasattr(self._parent, '_parent'):
            # its not root. The parent of the parent has a root attribute, where we can copy the
            # important values from
            self._set_child_node(self._parent)

    def _set_child_node(self, parent):
        """Set attributes of a child node need to be set by unpickling or copy"""
        if not hasattr(parent, self.SIGNALS[0]):
            # if the parent does not have a index field, we need to wait until the root is there
            # this will happen later
            return
        for sig in self.SIGNALS:
            setattr(self, sig, getattr(parent, sig))
        self._props.on_update.connect(self._on_dict_update)
        self.lock = parent.lock


# noinspection PyProtectedMember
class IndexingNode(Node):
    """ Extends the :class:`Node` for an additional property of indexing a specific
     property. It uses the signaling slots from the tree
    """
    __slots__ = ['index']

    # TODO: deletes... indexes of children?

    def __init__(self, name, parent=None, props=None, indexes=None,
                 props_cls=NotifyingDict):
        # pylint: disable=too-many-arguments
        super().__init__(name, parent, props, props_cls=props_cls)
        if parent is None and indexes:
            # this is the initalisation in the root

            #: the index to access a node quickly via a specific property, it has to be
            #: provided in the ctor of the root IndexingNode and is a dict consisting out
            #: of dicts, where the key of the first dict is the name of the index, and the
            #: second the index value itself.
            self.index = {index: {} for index in indexes}

            self._connect_events()

        elif parent is not None:
            self.index = parent.index
        else:
            raise ValueError("No index given")

    def _connect_events(self):
        """Wire up events to be able to groom the index."""
        self.on_update.connect(self._update_index, weak=False)
        self.on_create.connect(self._create_node, weak=False)
        self.on_delete.connect(self._on_node_delete, weak=False)

    def create_instance(self, name=None):
        """ a factory function which creates an instance with the same properties as
        this node, but not attached to it """
        return IndexingNode(name=None, indexes=list(self.index.keys()))

    def _create_node(self, node):
        """ created node """
        self._update_index(node, other=node.props)

    def _on_node_delete(self, node):
        """ event handler if node gets deleted: deletes all index properties for all
        children including itself. It replaces all signals by None objects """
        for iterated_node in node:

            for key, value in iterated_node.props.items():
                if key in self.index:
                    self.index[key].pop(value, None)
                    break

    def _update_index(self, node, other):
        """ This is the slot which updates the index """
        for key, new_value in other.items():
            if key not in self.index:
                continue

            old_value = node.props.get(key)

            if new_value == old_value and node is self.index[key].get(new_value):
                # if the value stays the some, do nothing
                continue

            if new_value is DELETE:
                del self.index[key][old_value]
                continue

            if new_value in self.index[key]:
                raise KeyError(
                    'The key "{}" already exists in index "{}"'.format(new_value, key))

            if old_value in self.index[key]:
                del self.index[key][old_value]

            self.index[key][new_value] = node

    def add_merge_set_with_id(self, iterator, key, using_update=False):
        """Similar to add_nodes.

        First try to merge by finding the parent using the provided `parent_id`
        If this fails, use the `

        The avaliable actions are `DELETE` and `UPSERT`

        :param iterator: (action, parent_id, name, props)
        :param key:
        :return:
        """
        # pylint: disable=too-many-branches

        def _handle_upsert(node, parent_node, name, props, using_update):
            logger.debug("Handling upsert (%s)", locals())
            if node is not None:
                if name is not None:
                    node.name = name

                node.parent = parent_node

                if using_update:
                    node.props.update(props)
                else:
                    node.props = props
            else:
                try:
                    node = parent_node.add_child(name, props)
                except DuplicateChildError:
                    logger.info('Duplicate child', exc_info=True)

            return node

        parent_map = {}
        # iterate through all of them, create a tree in the parent_map
        for item in iterator:
            operation, parent_id, name, props = item
            logger.debug('Merge attempt: %s for %s (%s props)', operation, name, 'updating' if using_update else 'overwriting')
            props = dict(props)
            try:
                node = self.index[key].get(props[key])
                if operation == 'DELETE':
                    if node is None:
                        # a delete to non exiting node, we don't care
                        continue
                    node.delete()
                    logger.debug('Node deleted')
                else:
                    parent_node = self.index[key][parent_id]
                    _handle_upsert(node, parent_node, name, props, using_update)

            # TODO: why are the test working when uncommenting that
            # except DuplicateChildError:
            #     logger.debug(
            #         'Node "%s" was not added as child. There is already a '
            #         'child with the same name in this node or same id'
            #         'in this tree', name)
            except KeyError as e:
                # means the parent does not exitst atm do that with our algo-magic
                logger.info('Merge failed for %s. Appending to parent_map: %s', name, parent_id)
                parent_map.setdefault(parent_id, []).append(item)

        working_stack = [self]

        # all already existing parents should be added to the `working_stack`
        for parent_id in parent_map:
            if parent_id in self.index[key]:
                working_stack.append(self.index[key][parent_id])

        logger.debug("Working stack: %s, parent_map: %s", working_stack, parent_map)

        while working_stack:
            logger.debug('Working stack contains %s items.', len(working_stack))
            parent_node = working_stack.pop()

            items = parent_map.pop(parent_node.props[key], [])
            logger.debug('Parent <%s> has %s children to merge.', parent_node.name, len(items))

            for item in items:
                operation, _, name, props = item
                props = dict(props)
                # get old node if there is one
                node = self.index[key].get(props[key])

                # DELETEs are always handled in the first pass, we only expect UPSERTs here
                assert operation == 'UPSERT'

                node = _handle_upsert(node, parent_node, name, props, using_update)

                if node is not None:
                    working_stack.append(node)

        # flatten out the values and return them
        return [item for sublist in parent_map.values() for item in sublist]

    def __getstate__(self):
        state = super().__getstate__()
        if self._parent is None:
            state['index'] = self.index
        return state

    def __setstate__(self, state):
        if 'index' in state:
            self.index = state['index']

        super().__setstate__(state)
        if self._parent is None:
            # the root need to get wired to its event handlers
            self._connect_events()

    def _set_child_node(self, parent):
        """Each node should have a reference to the index"""
        super()._set_child_node(parent)
        # if the parent does not have a index field, we need to wait until the root is there
        # this will happen later
        if hasattr(parent, 'index'):
            self.index = parent.index


def _tree_to_str_list(node, level=0, current=None, prop_key=None):   # pragma: no cover
    """Creates a stringlist prepared for printing.

    :param level: the depth of the current child.
    :param current: the list that has been generated by previous iterations.
    :param prop_key: an optional key to extract from each node.
    """
    if current is None:
        current = []
    for child in node.children:
        item = ('+-' * level) + child.name
        if prop_key is not None:
            item += ' -> [ {}: {} ]'.format(prop_key, child.props.get(prop_key))
        current.append(item)
        _tree_to_str_list(child, level + 1, current, prop_key=prop_key)
    return current


def tree_to_str(node, prop_key=None):    # pragma: no cover
    """Return a printable string of the tree's structure.

    :param node: the node at which to start.
    :param prop_key: an optional key to extract from each node.
    """
    lst = _tree_to_str_list(node, prop_key=prop_key)
    root_node = 'root' + ' -> [ {}: {} ]\n'.format(prop_key, node.props.get(prop_key))
    return root_node + '\n'.join(lst)


class DuplicateChildError(KeyError):
    """Error for duplicate children.

    Used mainly for id based storages which can have multiple files with the same name
    """
    pass


"""Namedtuple to keep track of the  merges to be applied to the tree."""
NodeChange = namedtuple('NodeChange', ['action', 'parent_id', 'name', 'props'])
