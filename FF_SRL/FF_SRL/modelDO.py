import warp as wp
import math
import numpy as np
import FF_SRL as dk
import torch
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade, Sdf, Vt, Gf

PI = wp.constant(math.pi)
ROTATIONCENTER_XFORM = wp.constant(3)
ROD_XFORM = wp.constant(0)
LEFTCLAMP_XFORM = wp.constant(4)
ACTION_ROD_FORWARD = wp.constant(0)
ACTION_ROD_ROTATE_CLOCKWISE = wp.constant(1)
ACTION_ROTATIONCENTER_X = wp.constant(2)
ACTION_ROTATIONCENTER_Y = wp.constant(3)
ACTION_CLAMP_OPEN = wp.constant(4)

CLAMP_DRAG_ANGLE = wp.constant(3)
CLAMP_MAX_ANGLE = wp.constant(85)
CLAMP_MIN_ANGLE = wp.constant(40)

in_de_crease_steps = 5


@wp.kernel
def setTransformKernelNotDO(xform: wp.array(dtype=wp.mat44),
                            xformId: wp.int32,
                            translationArray: wp.array(dtype=wp.vec3),
                            translationIndex:wp.int32):
    
    translation = translationArray[translationIndex]

    matrix = getTransformationMatrix(translation[0], translation[1], translation[2],
                                     0.0, 0.0, 0.0)
    
    xform[xformId] = wp.transpose(matrix)

@wp.kernel
def updateMeshKernelNotDO(src: wp.array(dtype=wp.vec3),
                          dest: wp.array(dtype=wp.vec3),
                          pointsShift: wp.int32,
                          xform: wp.array(dtype=wp.mat44),
                          xFormId: wp.int32):

    tid = wp.tid()

    tXForm = wp.transpose(xform[xFormId])

    p = src[tid]
    m = wp.transform_point(tXForm, p)

    dest[tid + pointsShift] = m

@wp.kernel
def transfromCartesianActionsKernel(cartesianActions: wp.array(dtype=wp.float32),
                                    laparoscopeTip: wp.array(dtype=wp.vec3),
                                    xform: wp.array(dtype=wp.mat44)):

    envNum = wp.tid()

    actionsIdA = 3 * envNum + 0
    actionsIdB = 3 * envNum + 1
    actionsIdC = 3 * envNum + 2

    if cartesianActions[actionsIdA] == 0.0 and cartesianActions[actionsIdB] == 0.0 and cartesianActions[actionsIdC] == 0.0:
        return

    laparoscopeRodXForm = xform[7 * envNum + ROD_XFORM]
    rotationCenterXForm = xform[7 * envNum + ROTATIONCENTER_XFORM]
    rotationCenterInvXFormT = wp.transpose(wp.inverse(rotationCenterXForm))

    currentEffectorPosWorld = laparoscopeTip[4 * envNum + 3]
    currentEffectorPosRelative = wp.transform_point(rotationCenterInvXFormT, currentEffectorPosWorld)
    delta = wp.transform_point(rotationCenterXForm, wp.vec3(cartesianActions[actionsIdA], cartesianActions[actionsIdB], cartesianActions[actionsIdC]))
    newEffectorPosRelative = currentEffectorPosRelative + delta

    radiusCurrent = wp.length(currentEffectorPosRelative)
    radiusNew = wp.length(newEffectorPosRelative)

    vectorCross = wp.cross(currentEffectorPosRelative, newEffectorPosRelative)
    sinTheta = wp.length(vectorCross)/(wp.length(currentEffectorPosRelative) * wp.length(newEffectorPosRelative))
    axis = wp.normalize(vectorCross)
    angle = wp.asin(sinTheta)

    transZero = wp.vec3()
    scale = wp.vec3(1.0, 1.0, 1.0)
    rotation = wp.quat_from_axis_angle(axis, angle)
    radius = radiusNew - radiusCurrent
    
    matrixLaparoscope = getTransformationMatrix(0.0, 0.0, radius, 0.0, 0.0, 0.0)
    transformFunc(xform, 7 * envNum + ROD_XFORM, wp.transpose(laparoscopeRodXForm), matrixLaparoscope)

    matrixRotation = wp.mat44(transZero, rotation, scale)
    transformFunc(xform, 7 * envNum + ROTATIONCENTER_XFORM, wp.transpose(rotationCenterXForm), matrixRotation)

@wp.kernel
def transfromCartesianActionsKernelInWorkspace(cartesianActions: wp.array(dtype=wp.float32),
                                               workspaceLow: wp.vec3,
                                               workspaceHigh: wp.vec3,
                                               laparoscopeTip: wp.array(dtype=wp.vec3),
                                               workspaceViolation: wp.array(dtype=wp.int32),
                                               xform: wp.array(dtype=wp.mat44)):

    envNum = wp.tid()

    actionsIdA = 3 * envNum + 0
    actionsIdB = 3 * envNum + 1
    actionsIdC = 3 * envNum + 2

    if cartesianActions[actionsIdA] == 0.0 and cartesianActions[actionsIdB] == 0.0 and cartesianActions[actionsIdC] == 0.0:
        return

    # Check if new laparoscope position within workspace
    workspaceViolation[envNum] = 0
    currentEffectorPosWorld = laparoscopeTip[4 * envNum + 3]
    cartesianActionsVec = wp.vec3(cartesianActions[actionsIdA], cartesianActions[actionsIdB], cartesianActions[actionsIdC])
    newEffectorPosWorld = currentEffectorPosWorld + cartesianActionsVec
    if newEffectorPosWorld[0] < workspaceLow[0] or newEffectorPosWorld[0] > workspaceHigh[0]:
        cartesianActionsVec[0] = 0.0
        workspaceViolation[envNum] = 1
    if newEffectorPosWorld[1] < workspaceLow[1] or newEffectorPosWorld[1] > workspaceHigh[1]:
        cartesianActionsVec[1] = 0.0
        workspaceViolation[envNum] = 1
    if newEffectorPosWorld[2] < workspaceLow[2] or newEffectorPosWorld[2] > workspaceHigh[2]:
        cartesianActionsVec[2] = 0.0
        workspaceViolation[envNum] = 1

    laparoscopeRodXForm = xform[7 * envNum + ROD_XFORM]
    rotationCenterXForm = xform[7 * envNum + ROTATIONCENTER_XFORM]
    rotationCenterInvXFormT = wp.transpose(wp.inverse(rotationCenterXForm))

    currentEffectorPosRelative = wp.transform_point(rotationCenterInvXFormT, currentEffectorPosWorld)
    delta = wp.transform_point(rotationCenterXForm, cartesianActionsVec)
    newEffectorPosRelative = currentEffectorPosRelative + delta

    radiusCurrent = wp.length(currentEffectorPosRelative)
    radiusNew = wp.length(newEffectorPosRelative)

    vectorCross = wp.cross(currentEffectorPosRelative, newEffectorPosRelative)
    sinTheta = wp.length(vectorCross)/(wp.length(currentEffectorPosRelative) * wp.length(newEffectorPosRelative))
    axis = wp.normalize(vectorCross)
    angle = wp.asin(sinTheta)

    transZero = wp.vec3()
    scale = wp.vec3(1.0, 1.0, 1.0)
    rotation = wp.quat_from_axis_angle(axis, angle)
    radius = radiusNew - radiusCurrent
    
    matrixLaparoscope = getTransformationMatrix(0.0, 0.0, radius, 0.0, 0.0, 0.0)
    transformFunc(xform, 7 * envNum + ROD_XFORM, wp.transpose(laparoscopeRodXForm), matrixLaparoscope)

    matrixRotation = wp.mat44(transZero, rotation, scale)
    transformFunc(xform, 7 * envNum + ROTATIONCENTER_XFORM, wp.transpose(rotationCenterXForm), matrixRotation)

@wp.kernel
def setRotationCenterTransformKernel(xform: wp.array(dtype=wp.mat44),
                                     translation: wp.array(dtype=wp.vec3),
                                     rotation: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    xFormId = tid*7 + ROTATIONCENTER_XFORM
    xForm = wp.transpose(xform[xFormId])

    # matrix = getTransformationMatrix(translation[tid * 3 + 0], translation[tid * 3 + 1], translation[tid * 3 + 2],
                                    #  rotation[tid * 3 + 0], rotation[tid * 3 + 1], rotation[tid * 3 + 2])
    matrix = getTransformationMatrix(translation[tid][0], translation[tid][1], translation[tid][2],
                                     rotation[tid][0], rotation[tid][1], rotation[tid][2])
    
    matrix = wp.mul(wp.inverse(xForm), matrix)

    transformFunc(xform, xFormId, xForm, matrix)
    
@wp.kernel
def applyActionsKernel(xform: wp.array(dtype=wp.mat44),
                       actions: wp.array(dtype=wp.float32),
                       laparoscopeDragCutHeat: wp.array(dtype=wp.int32),
                       laparoscopeDragCutHeatUpdate: wp.array(dtype=wp.int32)):
    
    tid = wp.tid()

    xForm = wp.transpose(xform[tid])
    numEnv = int(tid / 7)
    xFormKind = tid % 7

    if xFormKind == ROD_XFORM:
        matrix = getTransformationMatrix(0.0, 0.0, actions[5*numEnv], 0.0, 0.0, actions[5*numEnv + 1])
        transformFunc(xform, tid, xForm, matrix)
        return
    elif xFormKind == ROTATIONCENTER_XFORM:
        matrix = getTransformationMatrix(0.0, 0.0, 0.0, actions[5*numEnv + 2], actions[5*numEnv + 3], 0.0)
        transformFunc(xform, tid, xForm, matrix)
        return
    elif xFormKind == LEFTCLAMP_XFORM:
        rotateClampsFunc(xform, laparoscopeDragCutHeat, laparoscopeDragCutHeatUpdate, actions[5*numEnv + 4], numEnv)
        return
    else:
        return

@wp.kernel
def updateLaparoscopeMeshKernel(src: wp.array(dtype=wp.vec3),
                                dest: wp.array(dtype=wp.vec3),
                                xform: wp.array(dtype=wp.mat44),
                                vertexToXForm: wp.array(dtype=wp.int32),
                                visPointShift: wp.int32):

    tid = wp.tid()
    xFormId = vertexToXForm[tid]

    tXForm = wp.transpose(xform[xFormId])

    p = src[tid]
    m = wp.transform_point(tXForm, p)

    dest[tid + visPointShift] = m

@wp.kernel
def updateLaparoscopeKernel(xform: wp.array(dtype=wp.mat44),
                            laparoscopeTip: wp.array(dtype=wp.vec3),
                            laparoscopeBase: wp.array(dtype=wp.vec3),
                            laparoscopeRadius: wp.array(dtype=float),
                            laparoscopeHeights: wp.array(dtype=float)):

    tid = wp.tid()

    numEnv = int(tid / 4)
    xFormKind = tid % 4

    if xFormKind < 3:
        tXForm = wp.transpose(xform[numEnv*7 + xFormKind])
        height = laparoscopeHeights[numEnv*4 + xFormKind]
        # baseLocal = wp.vec3(0.0, 0.0, 0.0 - height/ 2.0)
        # tipLocal = wp.vec3(0.0, 0.0, 0.0 + height / 2.0)
        baseLocal = wp.vec3(0.0, 0.0, 0.0 - height/ 1.85)
        tipLocal = wp.vec3(0.0, 0.0, 0.0 + height / 1.85)
    else:
        tXForm = wp.transpose(xform[numEnv*7])
        rodHeight = laparoscopeHeights[numEnv*4]
        clampHeight = laparoscopeHeights[numEnv*4 + 2]
        baseLocal = wp.vec3(0.0, 0.0, 0.0 - rodHeight * 0.5 - clampHeight * 1.0)
        tipLocal = wp.vec3(0.0, 0.0, 0.0 + rodHeight * 0.5 + clampHeight * 1.0)

    laparoscopeBase[numEnv*4 + xFormKind] = wp.transform_point(tXForm, baseLocal)
    laparoscopeTip[numEnv*4 + xFormKind] = wp.transform_point(tXForm, tipLocal)

@wp.kernel
def updateClampsDragKernel(positions: wp.array(dtype=wp.vec3),
                           vertexToEnv: wp.array(dtype=wp.int32),
                           laparoscopeInfo: wp.array(dtype=wp.vec3),
                           dragConstraints: wp.array(dtype=wp.float32),
                           lookupRadius: float,
                           stateConstraint: wp.array(dtype=wp.int32),
                           updateConstraint: wp.array(dtype=wp.int32)):

    tid = wp.tid()
    numEnv = vertexToEnv[tid]

    dragId = numEnv * 3 + 0
    cutId = numEnv * 3 + 1
    heatId = numEnv * 3 + 2

    if updateConstraint[dragId] == 0:
        return

    if(stateConstraint[dragId] == 1):

        position = positions[tid]
        laparoscopeDragPoint = laparoscopeInfo[numEnv * 4 + 3]

        length = wp.length(position - laparoscopeDragPoint)

        if (length < lookupRadius):
            dragConstraints[tid] = 1.0

        else:
            dragConstraints[tid] = 0.0
    else:
        dragConstraints[tid] = 0.0

@wp.kernel
def forceClampsDragKernel(dragConstraints: wp.array(dtype=wp.float32),
                          vertexId: wp.int32,
                          envId: wp.array(dtype=wp.int32),
                          on: wp.float32,
                          numVertices: wp.int32):

    tid = wp.tid()
    dragConstraints[envId[tid] * numVertices + vertexId] = on

@wp.func
def transformFunc(xform: wp.array(dtype=wp.mat44), id:wp.int32, xForm:wp.mat44, matrix:wp.mat44) -> wp.int32:

    # Apply transformation to children
    applyTransformationToChildren(xform, matrix, id)
    
    xform[id] = wp.transpose(wp.mul(xForm, matrix))

    return 1

@wp.func
def rotateClampsFunc(xform: wp.array(dtype=wp.mat44), dragCutHeat: wp.array(dtype=wp.int32), dragCutHeatUpdate: wp.array(dtype=wp.int32), action: wp.float32, numEnv:wp.int32) -> wp.int32:
    
    rodXformId = numEnv*7
    leftClampXFormId = numEnv*7+4
    rightClampXFormId = numEnv*7+5

    dragId = numEnv * 3 + 0
    cutId = numEnv * 3 + 1
    heatId = numEnv * 3 + 2

    diffAngle = action*180.0/PI

    if diffAngle == 0.0:
        return 1

    rodXForm = wp.transpose(xform[rodXformId])
    rodInvXForm = wp.inverse(rodXForm)

    leftClampXForm = wp.transpose(xform[leftClampXFormId])
    rightClampXForm = wp.transpose(xform[rightClampXFormId])
    rightClampLocalXForm = wp.mul(rodInvXForm, rightClampXForm)

    currentAngle = wp.asin(rightClampLocalXForm[2][1])*180.0/PI
    applyRotation = 0.0

    # chceck if drag particles have been updated already
    if dragCutHeat[dragId] == 1:
        if dragCutHeatUpdate[dragId] == 1:
            dragCutHeatUpdate[dragId] = 0

    if action > 0.0:
        # Check if not exceeding max clamp angles
        if currentAngle < CLAMP_MAX_ANGLE:
            if currentAngle + diffAngle > CLAMP_MAX_ANGLE:
                diffAngle = float(CLAMP_MAX_ANGLE) - currentAngle
            applyRotation = 1.0
        # # Handle dragging
        if currentAngle > (CLAMP_DRAG_ANGLE + CLAMP_MIN_ANGLE) and dragCutHeat[dragId] == 1:
            dragCutHeat[dragId] = 0
            dragCutHeatUpdate[dragId] = 1
    else:
        # Check if not exceeding max clamp angles
        if currentAngle > CLAMP_MIN_ANGLE:
            if currentAngle + diffAngle < CLAMP_MIN_ANGLE:
               diffAngle = float(CLAMP_MIN_ANGLE) - currentAngle
            applyRotation = 1.0
        # # Handle dragging
        if currentAngle < (CLAMP_DRAG_ANGLE + CLAMP_MIN_ANGLE) and dragCutHeat[dragId] == 0:
            dragCutHeat[dragId] = 1
            dragCutHeatUpdate[dragId] = 1

    if applyRotation > 0.0:
        diffRad = diffAngle*PI/180.0
        matrix = getTransformationMatrix(0.0, 0.0, 0.0, diffRad, 0.0, 0.0)
        matrixInv = getTransformationMatrix(0.0, 0.0, 0.0, -diffRad, 0.0, 0.0)
        xform[leftClampXFormId] = wp.transpose(wp.mul(leftClampXForm, matrixInv))
        applyTransformationToChildren(xform, matrixInv, leftClampXFormId)
        xform[rightClampXFormId] = wp.transpose(wp.mul(rightClampXForm, matrix))
        applyTransformationToChildren(xform, matrix, rightClampXFormId)

    return 1

@wp.func
def applyTransformationToChildren(xform: wp.array(dtype=wp.mat44), matrix: wp.mat44, parentId: int) -> wp.int32:
    
    numEnv = int(parentId / 7)
    xFormKind = parentId % 7
    
    # todo: ugly but just for now
    if(xFormKind == 4):
            xform[numEnv*7 + 1] = applyTransformationToChild(xform, matrix, parentId, numEnv*7 + 1)
    if(xFormKind == 5):
            xform[numEnv*7 + 2] = applyTransformationToChild(xform, matrix, parentId, numEnv*7 + 2)
    if(xFormKind == 3):
        for i in range(7):
            if not i == xFormKind:
                xform[numEnv*7 + i] = applyTransformationToChild(xform, matrix, parentId, numEnv*7 + i)
    if(xFormKind == 0):
        for i in range(1, 7):
            if not i == 3:
                xform[numEnv*7 + i] = applyTransformationToChild(xform, matrix, parentId, numEnv*7 + i)

    return 1

@wp.func
def applyTransformationToChild(xform: wp.array(dtype=wp.mat44), matrix: wp.mat44, parentId: int, childId: int) -> wp.mat44:

    parentXForm = wp.transpose(xform[parentId])
    parentInvXForm = wp.inverse(parentXForm)
    childXForm = wp.transpose(xform[childId])

    # C'_gl = R'_gl * M * R'_gl^-1 * C_gl
    return wp.transpose(wp.mul(parentXForm, wp.mul(matrix, wp.mul(parentInvXForm, childXForm))))

@wp.func
def getTransformationMatrix(shiftX: float, shiftY: float, shiftZ: float,
                            angleX: float, angleY: float, angleZ: float) -> wp.mat44:

    trans = wp.vec3(shiftX, shiftY, shiftZ)
    transZero = wp.vec3()
    rotZero = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 0.0), 0.0)
    rotX = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angleX)
    rotY = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angleY)
    rotZ = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angleZ)
    scale = wp.vec3(1.0, 1.0, 1.0)
    
    matrixShift = wp.mat44(trans, rotZero, scale)
    matrixX = wp.mat44(transZero, rotX, scale)
    matrixY = wp.mat44(transZero, rotY, scale)
    matrixZ = wp.mat44(transZero, rotZ, scale)

    # y=RX*RY*RZ*T*x
    # return wp.mul(matrixX, wp.mul(matrixY, wp.mul(matrixZ, matrixShift)))
    # y=T*RX*RY*RZ*x
    return wp.mul(matrixShift, wp.mul(matrixX, wp.mul(matrixY, matrixZ)))

class SimLockBoxDO(dk.SimObject):

    def __init__(self, prim:Usd.Prim, device:str="cuda") -> None:
        super().__init__(prim, device)

    def lockArray(self, vecArray, array):

        for i in range(len(vecArray)):
            vec = vecArray[i]
            vecLocalBox = self.xForm.GetInverse().Transform(Gf.Vec3f(vec))

            if vecLocalBox[0] < 1.0 and vecLocalBox[0] > -1.0:
                if vecLocalBox[1] < 1.0 and vecLocalBox[1] > -1.0:
                    if vecLocalBox[2] < 1.0 and vecLocalBox[2] > -1.0:
                        array[i] = 0.0

class SimMeshDO(dk.SimObject):

    def __init__(self, prim:Usd.Prim, device:str="cuda:0", requiresGrad=True) -> None:
        super().__init__(prim, device)

        ## Global
        self.translationWP = None
        self.rotationWP = None
        self.requiresGrad = requiresGrad

        ## Mesh parameters
        self.density = None
        self.ksDistance = None
        self.ksVolume = None
        self.ksContact = None
        self.ksDrag = None
        self.volumeCompliance = None
        self.deviatoricCompliance = None
        self.breakFactor = None
        self.breakThreshold = None
        self.heatLoss = None
        self.heatConduction = None
        self.heatTransfer = None
        self.heatLimit = None
        self.heatSphereRadius = None
        self.heatQuantum = None

        ## Mesh mapping
        self.pointToVertex = None
        self.pointToVertexHost = None
        self.vertexToPoint = None
        self.vertexToPointHost = None
        self.faceToTriangle = None
        self.faceToTriangleHost = None
        self.triangleToFace = None
        self.triangleToFaceHost = None
        self.edgesToTriangles = None
        self.edgesToTrianglesHost = None
        self.tetrahedronNeighbors = None
        self.tetrahedronNeighborsHost = None
        self.triangleToTetrahedron = None
        self.triangleToTetrahedronHost = None
        self.tetrahedronToTriangle = None
        self.tetrahedronToTriangleHost = None
        self.vertexToEdge = None
        self.vertexToEdgeLen = None
        self.vertexToTetrahedron = None
        self.vertexToTetrahedronLen = None

        ## Mesh data
        self.meshVisPoints = None
        self.meshNormals = None
        self.meshVisFaces = None
        self.meshVisPointsColors = None

        # Points
        self.vertex = None
        self.numVertices = None
        self.visPoint = None
        self.numVisPoints = None
        self.normal = None
        self.dP = None
        self.predictedVertex = None
        self.outputVertex = None
        self.velocity = None
        self.inverseMass = None
        self.inverseMassHost = None

        # Edges
        self.numEdges = None
        self.edge = None
        self.edgeLambda = None
        self.edgeRestLength = None
        self.activeEdge = None
        self.flipEdge = None

        # Triangles/Faces
        self.numTris = None
        self.triangle = None
        self.hole = None

        self.numVisFaces = None
        self.visFace = None
        self.visHole = None

        # Tetrahedrons
        self.numTetrahedrons = None
        self.tetrahedron = None
        self.tetrahedronLambda = None
        self.tetrahedronRestVolume = None
        self.tetrahedronInverseMassNeoHookean = None
        self.tetrahedronInverseMassNeoHookeanHost = None
        self.tetrahedronInverseRestPositionNeoHookean = None
        self.neoHookeanLambda = None
        self.activeTetrahedron = None
        self.flipTetrahedron = None
        
        # Stable Neo-Hookean lambdas (separate for deviatoric and hydrostatic)
        self.stableNHDeviatoricLambda = None
        self.stableNHHydrostaticLambda = None

        # Constraints
        self.constraintsNumber = None
        self.activeDragConstraint = None
        self.flipDragConstraint = None

        # Rendering
        self.texCoords = None
        self.texIndices = None
        self.texture = None

        self.getMeshProperties()
        self.getMeshMapping()
        self.getMeshData()

    def getMeshProperties(self):
        self.density = self.prim.GetAttribute("param:density").Get()
        self.ksDistance = self.prim.GetAttribute("param:ksDistance").Get()
        self.ksVolume = self.prim.GetAttribute("param:ksVolume").Get()
        self.ksContact = self.prim.GetAttribute("param:ksContact").Get()
        self.ksDrag = self.prim.GetAttribute("param:ksDrag").Get()
        self.volumeCompliance = self.prim.GetAttribute("param:volumeCompliance").Get()
        self.deviatoricCompliance = self.prim.GetAttribute("param:deviatoricCompliance").Get()
        self.breakFactor = self.prim.GetAttribute("param:breakFactor").Get()
        self.breakThreshold = self.prim.GetAttribute("param:breakThreshold").Get()
        self.heatLoss = self.prim.GetAttribute("param:heatLoss").Get()
        self.heatConduction = self.prim.GetAttribute("param:heatConduction").Get()
        self.heatTransfer = self.prim.GetAttribute("param:heatTransfer").Get()
        self.heatLimit = self.prim.GetAttribute("param:heatLimit").Get()  
        self.heatSphereRadius = self.prim.GetAttribute("param:heatSphereRadius").Get()  
        self.heatQuantum = self.prim.GetAttribute("param:heatQuantum").Get()
        self.hasReduceData = self.prim.HasAttribute("param:hasReduceData")

    def getMeshMapping(self):
        meshPointToVertex = self.prim.GetAttribute("mapping:pointToVertex").Get()
        meshVertexToEdge = self.prim.GetAttribute("mapping:vertexToEdge").Get()
        meshVertexToEdgeLen = self.prim.GetAttribute("mapping:vertexToEdgeLen").Get()
        meshVertexToTetrahedron = self.prim.GetAttribute("mapping:vertexToTetrahedron").Get()
        meshVertexToTetrahedronLen = self.prim.GetAttribute("mapping:vertexToTetrahedronLen").Get()
        meshVertexToTriangle = self.prim.GetAttribute("mapping:vertexToTriangle").Get()
        meshVertexToTriangleLen = self.prim.GetAttribute("mapping:vertexToTriangleLen").Get()

        meshPointToVertex = np.array(meshPointToVertex)
        meshPointToVertex = meshPointToVertex.ravel()
        self.pointToVertex = meshPointToVertex
        self.vertexToEdge = np.array(meshVertexToEdge)
        self.vertexToEdgeLen = np.array(meshVertexToEdgeLen)
        self.vertexToTetrahedron = np.array(meshVertexToTetrahedron)
        self.vertexToTetrahedronLen = np.array(meshVertexToTetrahedronLen)
        self.vertexToTriangle = np.array(meshVertexToTriangle)
        self.vertexToTriangleLen = np.array(meshVertexToTriangleLen)

    def getMeshData(self):
        self.meshVisPoints = self.prim.GetAttribute("points").Get()
        self.meshNormals = self.prim.GetAttribute("normals").Get()
        meshVisFaces = self.prim.GetAttribute("faceVertexIndices").Get()
        meshVisFaces = np.array(meshVisFaces)
        self.meshVisFaces = meshVisFaces.ravel()
        self.meshVisPointsColors = self.prim.GetAttribute("primvars:displayColor").Get()
        meshVertices = self.prim.GetAttribute("extMesh:vertex").Get()
        meshTriangles = self.prim.GetAttribute("extMesh:triangle").Get()
        meshInverseMasses = self.prim.GetAttribute("extMesh:inverseMass").Get()
        meshEdges = self.prim.GetAttribute("extMesh:edge").Get()
        meshEdges = np.array(meshEdges)
        meshEdges = meshEdges.ravel()
        meshEdgesRestLengths = self.prim.GetAttribute("extMesh:edgeRestLength").Get()
        meshTetrahedrons = self.prim.GetAttribute("extMesh:elem").Get()
        meshTetrahedrons = np.array(meshTetrahedrons)
        meshTetrahedrons = meshTetrahedrons.ravel()
        meshTetrahedronsRestVolumes = self.prim.GetAttribute("extMesh:tetrahedronRestVolume").Get()
        meshTetrahedronsInverseMassesNeoHookean = self.prim.GetAttribute("extMesh:inverseMassNeoHookean").Get()
        
        # CRITICAL: USD's inverseRestPosition data is incorrect (negative determinants)
        # Recompute Q matrices from rest vertex positions, matching C++ implementation:
        # X = [v0-v3, v1-v3, v2-v3], Q = X^-1
        print(f"Recomputing Q matrices for {int(len(meshTetrahedrons)/4)} tetrahedra...")
        meshTetrahedronsInverseRestPositionsNeoHookean = []
        for i in range(0, len(meshTetrahedrons), 4):
            v0_idx = int(meshTetrahedrons[i])
            v1_idx = int(meshTetrahedrons[i+1])
            v2_idx = int(meshTetrahedrons[i+2])
            v3_idx = int(meshTetrahedrons[i+3])
            
            v0 = np.array(meshVertices[v0_idx], dtype=np.float64)
            v1 = np.array(meshVertices[v1_idx], dtype=np.float64)
            v2 = np.array(meshVertices[v2_idx], dtype=np.float64)
            v3 = np.array(meshVertices[v3_idx], dtype=np.float64)
            
            # Build rest shape matrix: X = [v0-v3, v1-v3, v2-v3]
            X = np.column_stack([v0 - v3, v1 - v3, v2 - v3])
            
            # Compute Q = X^-1
            Q = np.linalg.inv(X)
            meshTetrahedronsInverseRestPositionsNeoHookean.append(Q)
        
        meshTetrahedronsInverseRestPositionsNeoHookean = np.array(meshTetrahedronsInverseRestPositionsNeoHookean)
        print(f"Q matrix recomputation complete. Sample det(Q): {np.linalg.det(meshTetrahedronsInverseRestPositionsNeoHookean[0]):.6f}")

        self.numVertices = len(meshVertices)
        self.numVisPoints = len(self.meshVisPoints)
        self.numTris = int(len(meshTriangles) / 3)
        self.numVisFaces = int(len(meshVisFaces) / 3)
        self.numEdges = int(len(meshEdges) / 2)
        self.numTetrahedrons = int(len(meshTetrahedrons) / 4)

        # Save world to local transformation
        self.initialXForm = self.xForm

        # transform particles to world space
        self.vertex = self.transformLocalVecArray(meshVertices)
        self.visPoint = self.meshVisPoints

        # convert node inputs to GPU arrays
        self.triangle = meshTriangles
        self.visFace = meshVisFaces
        self.inverseMass = meshInverseMasses

        self.edge = meshEdges
        self.edgeRestLength = meshEdgesRestLengths

        self.tetrahedron = meshTetrahedrons
        self.tetrahedronRestVolume = meshTetrahedronsRestVolumes
        self.inverseRestPositions = meshTetrahedronsInverseRestPositionsNeoHookean

        # check if object has material
        bindingAPI = UsdShade.MaterialBindingAPI(self.prim)
        rel = bindingAPI.GetDirectBindingRel()
        pathList = rel.GetTargets()
        if len(pathList) > 0:
            primMat = self.prim.GetStage().GetPrimAtPath(pathList[0].pathString + "/Shader")

            # Some scenes bind materials without a usable Shader prim or without a diffuse texture.
            # Treat those meshes as untextured instead of crashing.
            if primMat and primMat.IsValid():
                diffuse_attr = primMat.GetAttribute("inputs:diffuse_texture")
                if diffuse_attr and diffuse_attr.HasAuthoredValue():
                    asset_path = diffuse_attr.Get()
                    if asset_path is not None and getattr(asset_path, "resolvedPath", None):
                        self.texture = asset_path.resolvedPath

            st_attr = self.prim.GetAttribute("primvars:st")
            if st_attr and st_attr.HasAuthoredValue():
                self.texCoords = st_attr.Get()

            st_idx_attr = self.prim.GetAttribute("primvars:st:indices")
            if st_idx_attr and st_idx_attr.HasAuthoredValue():
                self.texIndices = st_idx_attr.Get()
            elif self.texCoords is not None:
                self.texIndices = meshVisFaces

class SimRigidDO(dk.SimObject):

    def __init__(self, prim:Usd.Prim, device:str="cuda:0", requiresGrad=True) -> None:
        super().__init__(prim, device)

        ## Global
        self.translationWP = None
        self.rotationWP = None
        self.requiresGrad = requiresGrad

        ## Mesh data
        self.meshVisPoints = None
        self.meshNormals = None
        self.meshVisFaces = None
        self.meshVisPointsColors = None

        # Points
        self.vertex = None
        self.numVertices = None
        self.visPoint = None
        self.numVisPoints = None
        self.normal = None

        # Triangles/Faces
        self.numTris = None
        self.triangle = None
        self.hole = None

        self.numVisFaces = None
        self.visFace = None
        self.visHole = None

        # Rendering
        self.texCoords = None
        self.texIndices = None
        self.texture = None

        self.getData()

    def getData(self):

        self.meshVisPoints = self.prim.GetAttribute("points").Get()
        self.meshNormals = self.prim.GetAttribute("normals").Get()
        meshVisFaces = self.prim.GetAttribute("faceVertexIndices").Get()
        meshVisFaces = np.array(meshVisFaces)
        self.meshVisFaces = meshVisFaces.ravel()
        self.meshVisPointsColors = self.prim.GetAttribute("primvars:displayColor").Get()

        self.numVertices = len(self.meshVisPoints)
        self.numVisPoints = len(self.meshVisPoints)
        self.numTris = int(len(meshVisFaces) / 3)
        self.numVisFaces = int(len(meshVisFaces) / 3)

        # Save world to local transformation
        self.initialXForm = self.xForm

        # transform particles to world space
        self.vertex = self.transformLocalVecArray(self.meshVisPoints)
        # self.visPoint = self.meshVisPoints
        self.visPoint = self.vertex
        self.meshVisPoints = np.array(self.meshVisPoints)

        # convert node inputs to GPU arrays
        self.triangle = meshVisFaces
        self.visFace = meshVisFaces

        # check if object has material
        bindingAPI = UsdShade.MaterialBindingAPI(self.prim)
        rel = bindingAPI.GetDirectBindingRel()
        rel = bindingAPI.GetDirectBindingRel()
        pathList = rel.GetTargets()
        if len(pathList) > 0:
            pathList[0].pathString
            primMat = self.prim.GetStage().GetPrimAtPath(pathList[0].pathString + "/Shader")
            self.texture = primMat.GetAttribute("inputs:diffuse_texture").Get().resolvedPath
            self.texCoords = self.prim.GetAttribute("primvars:st").Get()
            if self.prim.GetAttribute("primvars:st:indices"):
                self.texIndices = self.prim.GetAttribute("primvars:st:indices").Get()
            else:
                self.texIndices = meshVisFaces

class SimConnectorDO(dk.SimObject):

    def __init__(self, prim: Usd.Prim, device: str = "cuda") -> None:
        super().__init__(prim, device)

        self.pointBodyPath = None
        self.triBodyPath = None

        self.massWeightRatio = None
        self.kS = None
        self.restLengthMul = None

        self.numTris = None

        self.pointIds = None
        self.triIds = None
        self.triBar = None
        self.restDist = None

        self.pointBodyPredictedVertex = None
        self.pointBodyDP = None
        self.pointBodyConstraintsNumber = None
        self.triBodyPredictedVertex = None
        self.triBodyDP = None
        self.triBodyConstraintsNumber = None

        self.updateConnector()

    def updateConnector(self) -> None:

        self.pointBodyPath = self.prim.GetRelationship("pointBody").GetTargets()[0]
        self.triBodyPath = self.prim.GetRelationship("triBody").GetTargets()[0]

        self.massWeightRatio = self.prim.GetAttribute("param:massWeightRatio").Get()
        self.kS = self.prim.GetAttribute("param:kS").Get()
        self.restLengthMul = self.prim.GetAttribute("param:restLengthMul").Get()

        pointIds = self.prim.GetAttribute("pointId").Get()
        triIds = np.array(self.prim.GetAttribute("triIds").Get()).ravel()
        triBar = self.prim.GetAttribute("triBar").Get()
        restDist = self.prim.GetAttribute("restDist").Get()

        self.numTris = int(len(triIds)/3)

        self.pointIds = pointIds
        self.triIds = triIds
        self.triBar = triBar
        self.restDist = restDist

class SimAdhesionDO():
    """
    Data class for adhesion bond topology, loaded from USD scene.

    Each adhesion bond connects a vertex on one deformable mesh (e.g., tumor)
    to a triangle on another mesh (e.g., bone surface).

    Uses UnifiedDistanceConstraint: C(d) = d - d*(d) with frozen barycentric coords.
    """

    def __init__(self, prim, device="cuda"):
        self.device = device

        # Read bond topology from USD attributes
        self.vertexBodyPath = prim.GetRelationship("vertexBody").GetTargets()[0] if prim.GetRelationship("vertexBody").GetTargets() else None
        self.triBodyPath = prim.GetRelationship("triBody").GetTargets()[0] if prim.GetRelationship("triBody").GetTargets() else None

        self.vertexIds = list(prim.GetAttribute("vertexIds").Get() or [])
        triIds = prim.GetAttribute("triIds").Get()
        self.triIds = list(np.array(triIds).ravel()) if triIds is not None else []

        triBar = prim.GetAttribute("triBar").Get()
        self.triBar = list(triBar) if triBar is not None else []

        restGap = prim.GetAttribute("restGap").Get()
        self.restGap = list(restGap) if restGap is not None else []

        self.numBonds = len(self.vertexIds)

        # Curve parameters (with defaults matching C++ InterDeformUnifiedDistanceConstraint)
        self.dContact = float(prim.GetAttribute("param:dContact").Get() or 0.0003)
        self.dRest = float(prim.GetAttribute("param:dRest").Get() or 0.0015)
        self.dNeutralStart = float(prim.GetAttribute("param:dNeutralStart").Get() or 0.003)
        self.breakRatio = float(prim.GetAttribute("param:breakRatio").Get() or 3.0)
        self.stretchAbsMin = float(prim.GetAttribute("param:stretchAbsMin").Get() or 0.005)
        self.alpha = float(prim.GetAttribute("param:alpha").Get() or 1e-5)


class SimModelDO():

    def __init__(self, stage, numEnvs, device,
                 simSubsteps=8,
                 simFrameRate=60,
                 simConstraintsSteps=1,
                 environmentGravity=[0.0, -9.8, 0.0],
                 environmentVelocityDampening=0.1,
                 environmentGroundLevel=0.0,
                 globalKsDistance=1.0,
                 globalKsVolume=1.0,
                 globalKsContact=1.0,
                 globalKsDrag=1.0,
                 globalVolumeCompliance=1.0,
                 globalDeviatoricCompliance=1.0,
                 globalBreakFactor=1.0,
                 globalBreakThreshold=1.0,
                 globalHeatLoss=1.0,
                 globalHeatConduction=1.0,
                 globalHeatTransfer=1.0,
                 globalHeatLimit=1.0,
                 globalHeatSphereRadius=1.0,
                 globalHeatQuantum=1.0,
                 globalLaparoscopeDragLookupRadius=0.15,
                 globalConnectorMassWeightRatio=0.75,
                 globalConnectorKs=0.125,
                 globalConnectorRestLengthMul=0.02,
                 # Adhesion parameters (UnifiedDistanceConstraint)
                 globalAdhesionDContact=0.0003,
                 globalAdhesionDRest=0.0015,
                 globalAdhesionDNeutralStart=0.003,
                 globalAdhesionBreakRatio=3.0,
                 globalAdhesionStretchAbsMin=0.005,
                 globalAdhesionAlpha=1e-5,
                 workspaceLow=[-1e5, -1e5, -1e5],
                 workspaceHigh=[1e5, 1e5, 1e5],
                 startingBoxLow=[-1e5, -1e5, -1e5],
                 startingBoxHigh=[1e5, 1e5, 1e5]) -> None:

        debugTimes = False
        with wp.ScopedTimer("Whole init", active=debugTimes, detailed=False):
            self.simBVHs = []
            self.stage = stage
            self.device = device
            self.numEnvs = numEnvs
            self.cartesianActions = 3

            self.simSubsteps = simSubsteps
            self.simFrameRate = simFrameRate
            self.simConstraints = simConstraintsSteps
            self.simDt = (1.0/self.simFrameRate)/self.simSubsteps

            self.workspaceLow = wp.vec3(workspaceLow)
            self.workspaceHigh = wp.vec3(workspaceHigh)
            self.startingBoxLowTH = startingBoxLow
            self.startingBoxHighTH = startingBoxHigh
            self.startingBoxLow = wp.vec3(self.startingBoxLowTH)
            self.startingBoxHigh = wp.vec3(self.startingBoxHighTH)

            self.gravity = wp.vec3(environmentGravity[0], environmentGravity[1], environmentGravity[2])
            self.velocityDampening = environmentVelocityDampening
            self.velocityDamping = 1.0  # Per-substep velocity multiplier (1.0 = no damping, 0.99 = light, 0.95 = heavy)
            self.groundLevel = environmentGroundLevel
            self.globalKsDistance = globalKsDistance
            # XPBD compliance α̃ = α/dt². α̃=0 → infinitely stiff (PBD kS=1).
            # α = 1/stiffness. Higher kS → stiffer → smaller compliance.
            # Map kS ∈ (0,1] → α̃: when kS=1, α̃≈0 (rigid); when kS→0, α̃→large (soft).
            self.globalDistanceCompliance = (1.0 - min(globalKsDistance, 0.999)) / (self.simDt * self.simDt)
            self.globalKsVolume = globalKsVolume
            self.globalVolumeCompliance = (1.0 - min(globalKsVolume, 0.999)) / (self.simDt * self.simDt)
            self.globalKsContact = globalKsContact
            self.globalKsDrag = globalKsDrag
            self.globalVolumeComplianceNeoHookean = globalVolumeCompliance
            self.gloabalDeviatoricComplianceNeoHookean = globalDeviatoricCompliance
            
            # Stable Neo-Hookean material parameters
            # Soft tissue: E=8e4 Pa, ν=0.45 → μ≈27.6kPa, λ≈248kPa
            # Scaled for cm units with unit masses (×1e6/Pa from meter-based values)
            self.globalMu = 1e7  # Shear modulus (cm-unit scaled)
            self.globalLambda = 5e7  # First Lamé parameter (cm-unit scaled)
            self.useStableNH = True  # Set to True to enable Stable Neo-Hookean constraints
            
            self.globalLaparoscopeDragLookupRadius = globalLaparoscopeDragLookupRadius

            self.globalBreakFactor = globalBreakFactor
            self.globalBreakThreshold = globalBreakThreshold
            self.globalHeatLoss = globalHeatLoss
            self.globalHeatConduction = globalHeatConduction
            self.globalHeatTransfer = globalHeatTransfer
            self.globalHeatLimit = globalHeatLimit
            self.globalHeatSphereRadius = globalHeatSphereRadius
            self.globalHeatQuantum = globalHeatQuantum

            self.globalConnectorMassWeightRatio = globalConnectorMassWeightRatio
            self.globalConnectorKs = globalConnectorKs
            self.globalConnectorRestLengthMul = globalConnectorRestLengthMul

            # Adhesion parameters (UnifiedDistanceConstraint curve)
            self.globalAdhesionDContact = globalAdhesionDContact
            self.globalAdhesionDRest = globalAdhesionDRest
            self.globalAdhesionDNeutralStart = globalAdhesionDNeutralStart
            self.globalAdhesionBreakRatio = globalAdhesionBreakRatio
            self.globalAdhesionStretchAbsMin = globalAdhesionStretchAbsMin
            self.globalAdhesionAlpha = globalAdhesionAlpha

            with wp.ScopedTimer("Setup SimEnvs", active=debugTimes, detailed=False):
                self.simEnvironment = dk.SimEnvironmentDO(self.stage, self.device, 0, globalLaparoscopeDragLookupRadius=self.globalLaparoscopeDragLookupRadius)

            self.environmentNumVertices = self.simEnvironment.environmentNumVertices

            self.reduce = False

            vertex = []
            visPoint = []
            visFace = []
            inverseMass = []
            triangle = []
            meshVisPointColors = []
            tetrahedron = None
            tetrahedronRestVolume = None
            inverseRestPositions = None
            edge = None
            edgeRestLength = None

            connectorPoints = []
            connectorTriangles = []
            connectorTriBar = []
            connectorRestDist = []

            adhesionVertexIds = []
            adhesionTriIds = []
            adhesionTriBar = []
            adhesionRestGap = []

            laparoscopeXForm = []
            laparoscopeBase = []
            laparoscopeRadius = []
            laparoscopeTip = []
            laparoscopeHeight = []
            laparoscopeVisPoint = []
            laparoscopeVisFace = []

            laparoscopeRodColor = None
            laparoscopeLeftClampColor = None
            laparoscopeRightClampColor = None
            laparoscopeVisPointColors = []

            rigidVisPoint = []
            rigidVisFace = []
            rigidVisPointColors = []

            pointToVertex = []
            triToEnv = []
            vertexToEnv = []
            laparoscopeVertexToXForm = []
            vertexToEdge = []
            vertexToEdgeLen = []
            vertexToEdgeStart = []
            vertexToTetrahedron = []
            vertexToTetrahedronLen = []
            vertexToTetrahedronStart = []
            vertexToTriangle = []
            vertexToTriangleLen = []
            vertexToTriangleStart = []

            currentVertices = 0
            currentTriangles = 0
            currentVisPoints = 0
            currentEdge = 0
            currentTetrahedron = 0

            currentRigidVertices = 0

            currentLaparoscopeVisPoints = 0

            with wp.ScopedTimer("Create lists", active=debugTimes, detailed=False):
                for i in range(self.numEnvs):
                    for j in range(len(self.simEnvironment.simMeshes)):
                        simMesh = self.simEnvironment.simMeshes[j]

                        vertex += simMesh.vertex
                        visPoint += list(simMesh.visPoint)

                        inverseMass += list(simMesh.inverseMass)

                        simTriangle = [x + currentVertices for x in simMesh.triangle]
                        simVisFace = [x + currentVisPoints for x in simMesh.visFace]
                        triangle += simTriangle
                        visFace += simVisFace
                        meshVisPointColors += list(simMesh.meshVisPointsColors)

                        if not tetrahedron is None:
                            tetrahedron = np.concatenate((tetrahedron, simMesh.tetrahedron + currentVertices))
                        else:
                            tetrahedron = simMesh.tetrahedron

                        if not tetrahedronRestVolume is None:
                            tetrahedronRestVolume = np.concatenate((tetrahedronRestVolume, simMesh.tetrahedronRestVolume))
                        else:
                            tetrahedronRestVolume = simMesh.tetrahedronRestVolume

                        if not inverseRestPositions is None:
                            inverseRestPositions = np.concatenate((inverseRestPositions, simMesh.inverseRestPositions))
                        else:
                            inverseRestPositions = simMesh.inverseRestPositions

                        if not edge is None:
                            edge = np.concatenate((edge, simMesh.edge + currentVertices))
                        else:
                            edge = simMesh.edge

                        if not edgeRestLength is None:
                            edgeRestLength = np.concatenate((edgeRestLength, simMesh.edgeRestLength))
                        else:
                            edgeRestLength = simMesh.edgeRestLength

                        simPointToVertex = list(simMesh.pointToVertex)
                        simPointToVertex[1::2] = [x + currentVertices for x in simPointToVertex[1::2]]
                        simPointToVertex[0::2] = [x + currentVisPoints for x in simPointToVertex[0::2]]
                        pointToVertex += simPointToVertex

                        if simMesh.hasReduceData:
                            
                            self.reduce = True

                            simVertexToEdge = list(simMesh.vertexToEdge)
                            simVertexToEdge = [x + currentEdge for x in simVertexToEdge]
                            vertexToEdge += simVertexToEdge

                            vertexToEdgeLen += list(simMesh.vertexToEdgeLen)

                            simVertexToTetrahedron = list(simMesh.vertexToTetrahedron)
                            simVertexToTetrahedron = [x + currentTetrahedron for x in simVertexToTetrahedron]
                            vertexToTetrahedron += simVertexToTetrahedron

                            vertexToTetrahedronLen += list(simMesh.vertexToTetrahedronLen)

                            simVertexToTriangle = list(simMesh.vertexToTriangle)
                            simVertexToTriangle = [x + currentTriangles for x in simVertexToTriangle]
                            vertexToTriangle += simVertexToTriangle

                            vertexToTriangleLen += list(simMesh.vertexToTriangleLen)

                        for k in range(len(self.simEnvironment.simConnectors)):

                            simConnector = self.simEnvironment.simConnectors[k]

                            if(simMesh.path == simConnector.pointBodyPath):

                                simConnectorPoints = [x + currentVertices for x in simConnector.pointIds]
                                connectorPoints += simConnectorPoints

                            if(simMesh.path == simConnector.triBodyPath):
                                simConnectorTriangles = [x + currentVertices for x in simConnector.triIds]
                                connectorTriangles += simConnectorTriangles

                        for k in range(len(self.simEnvironment.simAdhesions)):

                            simAdhesion = self.simEnvironment.simAdhesions[k]

                            if(simMesh.path == simAdhesion.vertexBodyPath):
                                simAdhesionVertexIds = [x + currentVertices for x in simAdhesion.vertexIds]
                                adhesionVertexIds += simAdhesionVertexIds

                            if(simMesh.path == simAdhesion.triBodyPath):
                                simAdhesionTriIds = [x + currentVertices for x in simAdhesion.triIds]
                                adhesionTriIds += simAdhesionTriIds

                        triToEnv += [i] * simMesh.numTris
                        vertexToEnv += [i] * simMesh.numVertices

                        currentVertices += simMesh.numVertices
                        currentTriangles += simMesh.numTris
                        currentVisPoints += simMesh.numVisPoints
                        currentEdge += simMesh.numEdges
                        currentTetrahedron += simMesh.numTetrahedrons

                    for j in range(len(self.simEnvironment.simRigids)):

                        simRigid = self.simEnvironment.simRigids[j]
                        rigidVisPoint += list(simRigid.vertex)

                        rigidVisFace += [x + currentRigidVertices for x in simRigid.triangle]
                        rigidVisPointColors += list(simRigid.meshVisPointsColors)

                        currentRigidVertices += simRigid.numVertices

                    for j in range(len(self.simEnvironment.simConnectors)):

                        simConnector = self.simEnvironment.simConnectors[j]

                        connectorTriBar += list(simConnector.triBar)
                        connectorRestDist += list(simConnector.restDist)

                    for j in range(len(self.simEnvironment.simAdhesions)):

                        simAdhesion = self.simEnvironment.simAdhesions[j]

                        adhesionTriBar += list(simAdhesion.triBar)
                        adhesionRestGap += list(simAdhesion.restGap)

                    for j in range(len(self.simEnvironment.simLaparoscopes)):

                        simLaparoscope = self.simEnvironment.simLaparoscopes[j]

                        laparoscopeXForm += simLaparoscope.laparoscopeXForm
                        laparoscopeBase += simLaparoscope.laparoscopeBase
                        laparoscopeTip += simLaparoscope.laparoscopeTip
                        laparoscopeHeight += simLaparoscope.laparoscopeHeight
                        laparoscopeRadius += simLaparoscope.laparoscopeRadius

                        if isinstance(simLaparoscope.visPoint, np.ndarray):
                            laparoscopeVisPoint += [tuple(v) for v in simLaparoscope.visPoint]
                        else:
                            laparoscopeVisPoint += simLaparoscope.visPoint
                        tmpVisFace = simLaparoscope.visFace
                        if isinstance(tmpVisFace, np.ndarray):
                            tmpVisFace = tmpVisFace.flatten().tolist()
                        laparoscopeVisFace += [x + currentLaparoscopeVisPoints for x in tmpVisFace]

                        laparoscopeVertexToXForm += [i * 7 + 6] * simLaparoscope.numRodVisPoint
                        laparoscopeVertexToXForm += [i * 7 + 4] * simLaparoscope.numLeftClampVisPoint
                        laparoscopeVertexToXForm += [i * 7 + 5] * simLaparoscope.numRightClampVisPoint

                        self.numRodVisPoint = simLaparoscope.numRodVisPoint
                        self.numLeftClampVisPoint = simLaparoscope.numLeftClampVisPoint
                        self.numRightClampVisPoint = simLaparoscope.numRightClampVisPoint

                        currentLaparoscopeVisPoints += simLaparoscope.numVisPoints

                        if simLaparoscope.hasLaparoscopeMesh:
                            laparoscopeVisPointColors += simLaparoscope.visPointColor
                        else:
                            laparoscopeRodColor = simLaparoscope.rodColor
                            laparoscopeLeftClampColor = simLaparoscope.leftClampColor
                            laparoscopeRightClampColor = simLaparoscope.rightClampColor

                            # Convert vec4 (RGBA) to vec3 (RGB) if needed
                            def _to_rgb(c):
                                return (float(c[0]), float(c[1]), float(c[2])) if len(c) > 3 else c

                            laparoscopeVisPointColors += [_to_rgb(laparoscopeRodColor)] * self.numRodVisPoint +\
                                                         [_to_rgb(laparoscopeLeftClampColor)] * self.numLeftClampVisPoint +\
                                                         [_to_rgb(laparoscopeRightClampColor)] * self.numRightClampVisPoint

            # Create variables that will be the same for all envs
            objectId = 0
            numVisFacesCummulative = 0
            textureId = 1
            objectUsesTexture = []
            objectNumVisFaceCummulative = []
            visFaceToObjectId = []
            textures = np.array([])
            texturesShift = []
            texturesSize = []
            texIndices = np.array([])
            texIndicesShift = []
            texCoords = np.array([])
            texCoordsShift = []

            # The order has to be maintained
            for j in range(len(self.simEnvironment.simMeshes)):
            # for j in range(1):
                simMesh = self.simEnvironment.simMeshes[j]
                if not simMesh.texture == None:
                    objectUsesTexture.append(textureId)
                    textureId += 1
                    texturesShift.append(int(len(textures)/3))
                    textures = np.append(textures, np.array(Image.open(simMesh.texture).convert('RGB')) / 255.0)
                    texturesSize.append(int(math.sqrt(int(len(textures)/3) - texturesShift[-1])))
                    texIndicesShift.append(int(len(texIndices)))
                    texIndices = np.append(texIndices, np.array(simMesh.texIndices))
                    texCoordsShift.append(int(len(texCoords)/2))
                    texCoords = np.append(texCoords, np.array(simMesh.texCoords))
                else:
                    objectUsesTexture.append(0)

                visFaceToObjectId.extend([objectId] * simMesh.numVisFaces)
                objectNumVisFaceCummulative.append(numVisFacesCummulative)
                numVisFacesCummulative += simMesh.numVisFaces
                objectId += 1

            for j in range(len(self.simEnvironment.simRigids)):
                simRigid = self.simEnvironment.simRigids[j]
                if not simRigid.texture == None:
                    objectUsesTexture.append(textureId)
                    textureId += 1
                    texturesShift.append(int(len(textures)/3))
                    textures = np.append(textures, np.array(Image.open(simRigid.texture).convert('RGB')) / 255.0)
                    texturesSize.append(int(math.sqrt(int(len(textures)/3) - texturesShift[-1])))
                    texIndicesShift.append(len(texIndices))
                    texIndices = np.append(texIndices, np.array(simRigid.texIndices))
                    texCoordsShift.append(int(len(texCoords)/2))
                    texCoords = np.append(texCoords, np.array(simRigid.texCoords))
                else:
                    objectUsesTexture.append(0)

                visFaceToObjectId.extend([objectId] * simRigid.numVisFaces)
                objectNumVisFaceCummulative.append(numVisFacesCummulative)
                numVisFacesCummulative += simRigid.numVisFaces
                objectId += 1

            for j in range(len(self.simEnvironment.simLaparoscopes)):
                simLaparoscope = self.simEnvironment.simLaparoscopes[j]
                if not simLaparoscope.texture == None:
                    objectUsesTexture.append(textureId)
                    textureId += 1
                    texturesShift.append(int(len(textures)/3))
                    textures = np.append(textures, np.array(Image.open(simLaparoscope.texture).convert('RGB')) / 255.0)
                    texturesSize.append(int(math.sqrt(int(len(textures)/3) - texturesShift[-1])))
                    texIndicesShift.append(len(texIndices))
                    texIndices = np.append(texIndices, np.array(simLaparoscope.texIndices))
                    texCoordsShift.append(int(len(texCoords)/2))
                    texCoords = np.append(texCoords, np.array(simLaparoscope.texCoords))
                else:
                    objectUsesTexture.append(0)

                visFaceToObjectId.extend([objectId] * simLaparoscope.numVisFaces)
                objectNumVisFaceCummulative.append(numVisFacesCummulative)
                numVisFacesCummulative += simLaparoscope.numVisFaces
                objectId += 1

            if self.reduce:
                startVertexEdge = 0
                startVertexTetrahedron = 0
                startVertexTriangle = 0
                for i in range(len(vertex)):
                   vertexToEdgeStart.append(startVertexEdge)
                   vertexToTetrahedronStart.append(startVertexTetrahedron)
                   vertexToTriangleStart.append(startVertexTriangle)
                   startVertexEdge += vertexToEdgeLen[i]
                   startVertexTetrahedron += vertexToTetrahedronLen[i]
                   startVertexTriangle += vertexToTriangleLen[i]

            with wp.ScopedTimer("Create warp arrays", active=debugTimes, detailed=False):
            # Vertices
                self.vertex = wp.array(vertex, dtype=wp.vec3, device=self.device, requires_grad=True)
                self.initialVertex = wp.array(vertex, dtype=wp.vec3, device=self.device)
                self.visPoint = wp.array(visPoint, dtype=wp.vec3, device=self.device)
                self.predictedVertex = wp.zeros_like(self.vertex, requires_grad=True)
                self.velocity = wp.zeros_like(self.vertex, requires_grad=True)
                self.initialVelocity = wp.zeros_like(self.vertex, requires_grad=True)
                self.dP = wp.zeros_like(self.vertex)
                self.constraintsNumber = wp.zeros(len(self.vertex), dtype=wp.int32, device=self.device)
                self.inverseMass = wp.array(inverseMass, dtype=wp.float32, device=self.device)

                # Triangles
                self.triangle = wp.array(triangle, dtype=wp.int32, device=self.device)
                self.visFace = wp.array(visFace, dtype=wp.int32, device=self.device)
                self.hole = wp.zeros(int(len(self.triangle)/3), dtype=wp.int32, device=self.device)

                # Tetrahedrons
                self.tetrahedron = wp.array(tetrahedron, dtype=wp.int32, device=self.device)
                self.activeTetrahedron = wp.array([1.0] * int(len(self.tetrahedron)/4), dtype=wp.float32, device=self.device)
                self.tetrahedronRestVolume = wp.array(tetrahedronRestVolume, dtype=wp.float32, device=self.device)
                self.tetrahedronDP = wp.zeros(len(self.activeTetrahedron) * 4, dtype=wp.vec3, device=self.device)
                
                # Stable Neo-Hookean lambda arrays (one per tetrahedron for each constraint type)
                self.stableNHDeviatoricLambda = wp.zeros(int(len(self.tetrahedron)/4), dtype=wp.float32, device=self.device)
                self.stableNHHydrostaticLambda = wp.zeros(int(len(self.tetrahedron)/4), dtype=wp.float32, device=self.device)
                
                # Inverse rest positions for Stable Neo-Hookean (Q matrix)
                if inverseRestPositions is not None:
                    self.tetrahedronInverseRestPositionNeoHookean = wp.array(inverseRestPositions, dtype=wp.mat33, device=self.device)
                else:
                    self.tetrahedronInverseRestPositionNeoHookean = None

                # Edges
                self.edge = wp.array(edge, dtype=wp.int32, device=self.device)
                self.activeEdge = wp.array([1.0] * int(len(self.edge)/2), dtype=wp.float32, device=self.device)
                self.edgeRestLength = wp.array(edgeRestLength, dtype=wp.float32, device=self.device)
                self.edgeDP = wp.zeros(len(self.activeEdge), dtype=wp.vec3, device=self.device)

                # XPBD: separate index arrays and lambda accumulators
                edge_np = np.array(edge, dtype=np.int32)
                n_edges = len(edge_np) // 2
                self.edgeA = wp.array(edge_np[0::2].copy(), dtype=wp.int32, device=self.device)
                self.edgeB = wp.array(edge_np[1::2].copy(), dtype=wp.int32, device=self.device)
                self.edgeLambda = wp.zeros(n_edges, dtype=wp.float32, device=self.device)

                tet_np = np.array(tetrahedron, dtype=np.int32)
                n_tets = len(tet_np) // 4
                self.tetrahedronA = wp.array(tet_np[0::4].copy(), dtype=wp.int32, device=self.device)
                self.tetrahedronB = wp.array(tet_np[1::4].copy(), dtype=wp.int32, device=self.device)
                self.tetrahedronC = wp.array(tet_np[2::4].copy(), dtype=wp.int32, device=self.device)
                self.tetrahedronD = wp.array(tet_np[3::4].copy(), dtype=wp.int32, device=self.device)
                self.volumeLambda = wp.zeros(n_tets, dtype=wp.float32, device=self.device)

                # Connectors
                self.connectorVertexId = wp.array(connectorPoints, dtype=wp.int32, device=self.device)
                self.connectorTriangleId = wp.array(connectorTriangles, dtype=wp.int32, device=self.device)
                self.connectorTriBar = wp.array(connectorTriBar, dtype=wp.vec3, device=self.device)
                self.connectorRestDist = wp.array(connectorRestDist, dtype=wp.float32, device=self.device)

                # Adhesion bonds (UnifiedDistanceConstraint)
                self.numAdhesionBonds = len(adhesionVertexIds)
                self.numAdhesionBondsPerEnv = len(adhesionVertexIds) // max(self.numEnvs, 1) if len(adhesionVertexIds) > 0 else 0
                if self.numAdhesionBonds > 0:
                    self.adhesionVertexId = wp.array(adhesionVertexIds, dtype=wp.int32, device=self.device)
                    self.adhesionTriId = wp.array(adhesionTriIds, dtype=wp.int32, device=self.device)
                    self.adhesionTriBar = wp.array(adhesionTriBar, dtype=wp.vec3, device=self.device)
                    self.adhesionRestGap = wp.array(adhesionRestGap, dtype=wp.float32, device=self.device)
                    self.adhesionActive = wp.array([1.0] * self.numAdhesionBonds, dtype=wp.float32, device=self.device)
                    self.adhesionInitialActive = wp.array([1.0] * self.numAdhesionBonds, dtype=wp.float32, device=self.device)
                    self.adhesionNormalCache = wp.zeros(self.numAdhesionBonds, dtype=wp.vec3, device=self.device)
                    self.adhesionCacheValid = wp.zeros(self.numAdhesionBonds, dtype=wp.int32, device=self.device)
                else:
                    self.adhesionVertexId = None
                    self.adhesionTriId = None
                    self.adhesionTriBar = None
                    self.adhesionRestGap = None
                    self.adhesionActive = None
                    self.adhesionInitialActive = None
                    self.adhesionNormalCache = None
                    self.adhesionCacheValid = None

                # Rigid-deform adhesion bonds (rigid body point fixed, deformable triangle moves)
                # These are created programmatically via createRigidAdhesionBondsProgrammatic()
                self.numRigidAdhesionBonds = 0
                self.numRigidAdhesionBondsPerEnv = 0
                self.rigidAdhesionRigidPoint = None
                self.rigidAdhesionTriId = None
                self.rigidAdhesionTriBar = None
                self.rigidAdhesionRestGap = None
                self.rigidAdhesionActive = None
                self.rigidAdhesionInitialActive = None
                self.rigidAdhesionNormalCache = None
                self.rigidAdhesionCacheValid = None

                # Miscelanious
                self.activeDragConstraint = wp.zeros(len(self.vertex), dtype=wp.float32, device=self.device)
                self.numVertices = len(self.vertex)
                self.numVisPoints = len(visPoint)
                self.numTetrahedrons = int(len(self.tetrahedron)/4)
                self.numTriangles = int(len(self.triangle)/3)
                self.numVisFaces = int(len(visFace)/3)
                self.numEdges = int(len(self.edge)/2)
                self.simConnectors = len(self.connectorVertexId)

                self.numRigidVisPoints = len(rigidVisPoint)
                self.numRigidVisFaces = int(len(rigidVisFace)/3)

                # Mapping
                self.pointToVertex = wp.array(pointToVertex, dtype=wp.int32, device=self.device)
                self.triToEnv = wp.array(triToEnv, dtype=wp.int32, device=self.device)
                self.vertexToEnv = wp.array(vertexToEnv, dtype=wp.int32, device=self.device)
            
                if self.reduce:
                    self.vertexToEdge = wp.array(vertexToEdge, dtype=wp.int32, device=self.device)
                    self.vertexToEdgeLen = wp.array(vertexToEdgeLen, dtype=wp.int32, device=self.device)
                    self.vertexToEdgeStart = wp.array(vertexToEdgeStart, dtype=wp.int32, device=self.device)
                    self.vertexToTetrahedron = wp.array(vertexToTetrahedron, dtype=wp.int32, device=self.device)
                    self.vertexToTetrahedronLen = wp.array(vertexToTetrahedronLen, dtype=wp.int32, device=self.device)
                    self.vertexToTetrahedronStart = wp.array(vertexToTetrahedronStart, dtype=wp.int32, device=self.device)
                    self.vertexToTriangle = wp.array(vertexToTriangle, dtype=wp.int32, device=self.device)
                    self.vertexToTriangleLen = wp.array(vertexToTriangleLen, dtype=wp.int32, device=self.device)
                    self.vertexToTriangleStart = wp.array(vertexToTriangleStart, dtype=wp.int32, device=self.device)

                # Laparoscope
                self.laparoscopeXForm = wp.array(laparoscopeXForm, dtype=wp.mat44, device=self.device)
                self.laparoscopeInitialXForm = wp.array(laparoscopeXForm, dtype=wp.mat44, device=self.device)
                self.laparoscopeBase = wp.array(laparoscopeBase, dtype=wp.vec3, device=self.device)
                self.laparoscopeInitialBase = wp.array(laparoscopeBase, dtype=wp.vec3, device=self.device)
                self.laparoscopeTip = wp.array(laparoscopeTip, dtype=wp.vec3, device=self.device)
                self.laparoscopeInitialTip = wp.array(laparoscopeTip, dtype=wp.vec3, device=self.device)
                self.laparoscopeRadius = wp.array(laparoscopeRadius, dtype=float, device=self.device)
                self.laparoscopeHeight = wp.array(laparoscopeHeight, dtype=float, device=self.device)
                self.laparoscopeDragCutHeat = wp.zeros(3 * self.numEnvs, dtype=wp.int32, device=self.device)
                self.laparoscopeDragCutHeatUpdate = wp.zeros(3 * self.numEnvs, dtype=wp.int32, device=self.device)

                self.laparoscopeVertex = wp.array(laparoscopeVisPoint, dtype=wp.vec3, device=self.device)
                self.laparoscopeVisPoint = wp.array(laparoscopeVisPoint, dtype=wp.vec3, device=self.device)
                self.laparoscopeVisFace = wp.array(laparoscopeVisFace, dtype=wp.int32, device=self.device)
                self.laparoscopeInitialVertex = wp.array(laparoscopeVisPoint, dtype=wp.vec3, device=self.device)
                self.laparoscopeVertexToXForm = wp.array(laparoscopeVertexToXForm, dtype=wp.int32, device=self.device)
                self.laparoscopeDP = wp.zeros(self.numTriangles * 3, dtype=wp.vec3, device=self.device)

                self.laparoscopeWorkspaceViolation = wp.zeros(self.numEnvs, dtype=wp.int32, device=self.device)
                self.laparoscopeInCollision = wp.zeros(self.numEnvs, dtype=wp.int32, device=self.device)

                self.numLaparoscopeVertices = len(self.laparoscopeVertex)
                self.numLaparoscopeVisFaces = int(len(laparoscopeVisFace) / 3.0)
                self.numLaparoscopeVisPoints = len(laparoscopeVisPoint)

                # Combined for all envs
                allVisPoint = visPoint + rigidVisPoint + laparoscopeVisPoint
                self.allVisPoint = wp.array(allVisPoint, dtype=wp.vec3, device=self.device)
                self.allVisPointInitial = wp.array(allVisPoint, dtype=wp.vec3, device=self.device)

                rigidFace = [x + currentVisPoints for x in rigidVisFace]
                laparoscopeFace = [x + currentVisPoints + currentRigidVertices for x in laparoscopeVisFace]
                allVisFace = visFace + rigidFace + laparoscopeFace

                self.allVisFace = wp.array(allVisFace, dtype=wp.int32, device=self.device)
                visVertexColor = meshVisPointColors + rigidVisPointColors + laparoscopeVisPointColors

                self.allVisPointColor = wp.array(visVertexColor, dtype=wp.vec3, device=self.device)

                self.numAllVisFaces = int(len(allVisFace) / 3.0)
                self.numAllVisPoints = len(allVisPoint)

            with wp.ScopedTimer("Create BVH data", active=debugTimes, detailed=False):
                self.createDataForBVH()
            # Needs to run, otherwise laparoscope parts are overlapped in 0, 0, 0 coords. and BVH is problematic

            with wp.ScopedTimer("Others", active=debugTimes, detailed=False):
                self.transformSimLaparoscopeMeshData()
                self.transformSimMeshData()

                # Rendering
                self.visFaceToObjectId = wp.array(visFaceToObjectId, dtype=wp.int32, device=self.device)
                self.objectNumVisFaceCummulative = wp.array(objectNumVisFaceCummulative, dtype=wp.int32, device=self.device)
                self.objectUsesTexture = wp.array(objectUsesTexture, dtype=wp.int32, device=self.device)
                self.textures = wp.array(textures, dtype=wp.vec3, device=self.device)
                self.texturesShift = wp.array(texturesShift, dtype=wp.int32, device=self.device)
                self.texturesSize = wp.array(texturesSize, dtype=wp.int32, device=self.device)
                self.texCoords = wp.array(texCoords, dtype=wp.vec2, device=self.device)
                self.texCoordsShift = wp.array(texCoordsShift, dtype=wp.int32, device=self.device)
                self.texIndices = wp.array(texIndices, dtype=wp.int32, device=self.device)
                self.texIndicesShift = wp.array(texIndicesShift, dtype=wp.int32, device=self.device)

    def createDataForBVH(self):
        #This is based on a fact that all environments are identical and will only store one BVH for all envs

        envVisPoint = []
        envVisFace = []
        currentVisPoints = 0
        currentVertices = 0

        self.numEnvMeshesVisPoints = 0
        self.numEnvMeshesVertices = 0
        self.numEnvMeshesVisFaces = 0
        self.numEnvRigidsVisPoints = 0
        self.numEnvRigidsVisFaces = 0
        self.numEnvLaparoscopeVisPoints = 0
        self.numEnvLaparoscopeVertices = 0
        self.numEnvLaparoscopeVisFaces = 0
        self.numEnvAllVisPoints = 0
        self.numEnvAllVisFaces = 0
        self.numEnvAllVertices = 0

        for j in range(len(self.simEnvironment.simMeshes)):
            simMesh = self.simEnvironment.simMeshes[j]

            envVisPoint += list(simMesh.visPoint)
            simVisFace = [x + currentVisPoints for x in simMesh.visFace]
            envVisFace += simVisFace
            currentVisPoints = len(envVisPoint)
            currentVertices += len(simMesh.vertex)

        self.numEnvMeshesVisPoints = currentVisPoints
        self.numEnvMeshesVertices = currentVertices
        self.numEnvMeshesVisFaces = int(len(envVisFace) / 3.0)

        for j in range(len(self.simEnvironment.simRigids)):
            simRigid = self.simEnvironment.simRigids[j]

            envVisPoint += list(simRigid.vertex)
            simVisFace = [x + currentVisPoints for x in simRigid.triangle]
            envVisFace += simVisFace
            currentVisPoints = len(envVisPoint)

        self.numEnvRigidsVisPoints = currentVisPoints - self.numEnvMeshesVisPoints
        self.numEnvRigidsVisFaces = int(len(envVisFace) / 3.0) - self.numEnvMeshesVisFaces

        for j in range(len(self.simEnvironment.simLaparoscopes)):
            simLaparoscope = self.simEnvironment.simLaparoscopes[j]

            if isinstance(simLaparoscope.visPoint, np.ndarray):
                envVisPoint += [tuple(v) for v in simLaparoscope.visPoint]
            else:
                envVisPoint += simLaparoscope.visPoint
            laparoVisFace = simLaparoscope.visFace
            if isinstance(laparoVisFace, np.ndarray):
                laparoVisFace = laparoVisFace.flatten().tolist()
            simVisFace = [x + currentVisPoints for x in laparoVisFace]
            envVisFace += simVisFace
            currentVisPoints = len(envVisPoint)

        self.numEnvAllVisPoints = currentVisPoints
        self.numEnvAllVisFaces = int(len(envVisFace) / 3.0)
        self.numEnvAllVertices = currentVertices

        self.numEnvLaparoscopeVisPoints = currentVisPoints - (self.numEnvMeshesVisPoints + self.numEnvRigidsVisPoints)
        self.numEnvLaparoscopeVisFaces = self.numEnvAllVisFaces - (self.numEnvMeshesVisFaces + self.numEnvRigidsVisFaces)

        self.envVisPoint = wp.array(envVisPoint, dtype=wp.vec3, device=self.device)
        self.envVisFace = wp.array(envVisFace, dtype=wp.int32, device=self.device)

    def applyActions(self, actions):

        wp.launch(kernel=applyActionsKernel,
                  dim=len(self.laparoscopeXForm),
                  inputs=[self.laparoscopeXForm,
                          actions,
                          self.laparoscopeDragCutHeat,
                          self.laparoscopeDragCutHeatUpdate],
                  device=self.device)

        wp.launch(kernel=updateLaparoscopeKernel, 
                  dim=4*self.numEnvs, 
                  inputs=[self.laparoscopeXForm,
                          self.laparoscopeTip,
                          self.laparoscopeBase,
                          self.laparoscopeRadius,
                          self.laparoscopeHeight],
                  device=self.device)
        
        wp.launch(kernel=updateClampsDragKernel, 
                  dim=self.numVertices, 
                  inputs=[self.predictedVertex,
                          self.vertexToEnv,
                          self.laparoscopeTip,
                          self.activeDragConstraint,
                          self.globalLaparoscopeDragLookupRadius,
                          self.laparoscopeDragCutHeat,
                          self.laparoscopeDragCutHeatUpdate],
                  device=self.device)
            
        self.transformSimMeshData()
        self.transformSimLaparoscopeMeshData()
    
    def applyCartesianActions(self, cartesianActions):

        wp.launch(kernel=transfromCartesianActionsKernel,
                  dim=self.numEnvs,
                  inputs=[cartesianActions,
                          self.laparoscopeTip,
                          self.laparoscopeXForm],
                  device=self.device)
        
        wp.launch(kernel=updateLaparoscopeKernel, 
                  dim=4*self.numEnvs, 
                  inputs=[self.laparoscopeXForm,
                          self.laparoscopeTip,
                          self.laparoscopeBase,
                          self.laparoscopeRadius,
                          self.laparoscopeHeight],
                  device=self.device)
            
        self.transformSimMeshData()
        self.transformSimLaparoscopeMeshData()
        
    def applyCartesianActionsInWorkspace(self, cartesianActions):

        wp.launch(kernel=transfromCartesianActionsKernelInWorkspace,
                  dim=self.numEnvs,
                  inputs=[cartesianActions,
                          self.workspaceLow,
                          self.workspaceHigh,
                          self.laparoscopeTip,
                          self.laparoscopeWorkspaceViolation,
                          self.laparoscopeXForm],
                  device=self.device)
        
        wp.launch(kernel=updateLaparoscopeKernel, 
                  dim=4*self.numEnvs, 
                  inputs=[self.laparoscopeXForm,
                          self.laparoscopeTip,
                          self.laparoscopeBase,
                          self.laparoscopeRadius,
                          self.laparoscopeHeight],
                  device=self.device)

    def setEffectorInitialPosition(self, cartesianActions):

        self.applyCartesianActions(cartesianActions)
        
        wp.copy(self.laparoscopeInitialXForm, self.laparoscopeXForm)
        wp.copy(self.laparoscopeInitialBase, self.laparoscopeBase)
        wp.copy(self.laparoscopeInitialTip, self.laparoscopeTip)

        wp.copy(self.laparoscopeInitialVertex, self.laparoscopeVertex)

    def setRotationCenterInitialPosition(self, laparoscopeTranslation, laparoscopeRotation):

        self.setRotationCenter(laparoscopeTranslation, laparoscopeRotation)
        wp.copy(self.laparoscopeInitialXForm, self.laparoscopeXForm)

    # not finished - no good way to do it now
    def setRigidTransformation(self, rigidId, positionArray, positionIndex):
        
        wp.launch(kernel=setTransformKernelNotDO,
                  dim=1,
                  inputs=[self.rigidXForm,
                          rigidId,
                          positionArray,
                          positionIndex,],
                  device=self.device)

        wp.launch(kernel=updateMeshKernelNotDO,
                  dim=self.numRigidVisPoints,
                  inputs=[self.rigidVertex,
                          self.allVisPoint,
                          self.numVisPoints,
                          self.rigidXForm,
                          rigidId],
                  device=self.device)

    def transformSimMeshData(self):
        dk.launchRemapAToB(self.visPoint, self.vertex, self.pointToVertex, self.device)
        dk.launchRemapAToBLimited(self.allVisPoint, self.vertex, self.pointToVertex, self.numVisPoints, self.device)
        dk.launchRemapAToBLimited(self.envVisPoint, self.vertex, self.pointToVertex, self.numEnvMeshesVisPoints, self.device)

    def transformSimLaparoscopeMeshData(self):

        wp.launch(kernel=updateLaparoscopeMeshKernel, 
                  dim=self.numLaparoscopeVertices, 
                  inputs=[self.laparoscopeVertex,
                          self.laparoscopeVisPoint,
                          self.laparoscopeXForm,
                          self.laparoscopeVertexToXForm,
                          0],
                  device=self.device)

        wp.launch(kernel=updateLaparoscopeMeshKernel, 
                  dim=self.numLaparoscopeVertices, 
                  inputs=[self.laparoscopeVertex,
                          self.allVisPoint,
                          self.laparoscopeXForm,
                          self.laparoscopeVertexToXForm,
                          (self.numEnvMeshesVisPoints + self.numEnvRigidsVisPoints)*self.numEnvs],
                  device=self.device)

        wp.launch(kernel=updateLaparoscopeMeshKernel, 
                  dim=self.numEnvLaparoscopeVisPoints, 
                  inputs=[self.laparoscopeVertex,
                          self.envVisPoint,
                          self.laparoscopeXForm,
                          self.laparoscopeVertexToXForm,
                          self.numEnvMeshesVisPoints + self.numEnvRigidsVisPoints],
                  device=self.device)

    def duplicateVisibleObjects(self, spacingX:float=3.0, spacingZ:float=3.0):
        for simEnvironment in self.simEnvironments:
            simEnvironment.removeOriginalMeshes()
            simEnvironment.removeOriginalLaparoscopes()

            simEnvironment.duplicateMeshes(spacingX, spacingZ)
            simEnvironment.duplicateLaparoscopes(spacingX, spacingZ)

    def getVertexPositionsTensor(self, vertexId):

        vertexTensor = wp.to_torch(self.vertex)

        return vertexTensor[vertexId]
    
    def getVertexPositionsTensorMultiEnvs(self, vertexIds):

        vertexTensor = wp.to_torch(self.vertex, requires_grad=False)
        indices = [vertexIds[i] + self.numEnvAllVertices*i for i in range(self.numEnvs)]

        return vertexTensor[indices]
    
    def getLaparoscopePositionsTensor(self):

        tipTensor = wp.to_torch(self.laparoscopeTip)

        return tipTensor[3 : : 4]
    
    def getModelObservationsTensor(self):
        """
        Get laparoscope observations for all environments as a single tensor.
        Returns tensor of shape [num_envs, 6] containing base and tip positions.
        Used by RL environment wrapper.
        """
        baseTensor = wp.to_torch(self.laparoscopeBase, requires_grad=False)
        tipTensor = wp.to_torch(self.laparoscopeTip, requires_grad=False)
        
        # Extract effector (tip) positions for each environment
        # laparoscopeTip has shape [num_envs * 4, 3], we want indices [3, 7, 11, ...]
        tipPositions = tipTensor[3::4]  # Shape: [num_envs, 3]
        
        # Also get base positions if needed
        basePositions = baseTensor[3::4]  # Shape: [num_envs, 3]
        
        # Concatenate base and tip for full observation
        # Shape: [num_envs, 6]
        observations = torch.cat([basePositions, tipPositions], dim=1)
        
        return observations
    
    def getModelRewardTensor(self):
        """
        Get mesh vertex positions for all environments for reward calculation.
        Returns tensor containing all vertex positions.
        Used by RL environment wrapper.
        """
        vertexTensor = wp.to_torch(self.vertex, requires_grad=False)
        return vertexTensor
    
    def getLaparoscopeClampVertex(self, vertexId):

        dragTensor = wp.to_torch(self.activeDragConstraint)

        return dragTensor[vertexId]    
    
    def getLaparoscopeWorkspaceViolation(self):

        laparoscopeWorkspaceViolation = wp.to_torch(self.laparoscopeWorkspaceViolation)

        return laparoscopeWorkspaceViolation
    
    def getLaparoscopeInCollision(self):

        laparoscopeInCollision = wp.to_torch(self.laparoscopeInCollision)

        return laparoscopeInCollision
    
    def resetCollisionInfo(self):

        self.laparoscopeInCollision = wp.zeros(self.numEnvs, dtype=wp.int32, device=self.device)
    
    def forceLaparoscopeClamp(self, envs):
        envs = torch.clamp(envs, 0., 1.)
        action = torch.tensor([0., 0., 0., 0., -1.], dtype=torch.float, device=self.device)
        actions = action.repeat(self.numEnvs, 1) * torch.transpose(envs.repeat(5, 1), 0, 1)
        actions = actions.flatten()

        # Need to apply twice to check for dragging
        self.applyActions(wp.from_torch(actions))
         
    def forceLaparoscopeClampRegion(self, centerVertexId: int, radius: float, envs, on: float = 1.0, animate: bool = True):
        """Grab all vertices within `radius` (cm) of centerVertexId."""
        verts = self.vertex.numpy()
        inv_mass = self.inverseMass.numpy()
        center_pos = verts[centerVertexId]

        # Find all free vertices within radius (single-env indices)
        n_env_verts = self.numEnvAllVertices
        grabbed = []
        for i in range(n_env_verts):
            if inv_mass[i] == 0.0:
                continue
            dist = np.linalg.norm(verts[i] - center_pos)
            if dist <= radius:
                grabbed.append(i)

        print(f"  forceLaparoscopeClampRegion: {len(grabbed)} vertices within {radius:.2f}cm of vertex {centerVertexId}")

        # Set drag constraint for each grabbed vertex in each env
        envIds = torch.nonzero(envs).to(dtype=torch.int32).flatten()
        envIds_wp = wp.from_torch(envIds)
        for vid in grabbed:
            wp.launch(kernel=forceClampsDragKernel,
                      dim=len(envIds_wp),
                      inputs=[self.activeDragConstraint,
                              vid,
                              envIds_wp,
                              on,
                              self.environmentNumVertices],
                      device=self.device)

        if animate:
            action = torch.tensor([0., 0., 0., 0., -1.], dtype=torch.float, device=self.device)
            actions = action.repeat(self.numEnvs, 1) * torch.transpose(envs.repeat(5, 1), 0, 1)
            actions = actions.flatten()
            actions = wp.from_torch(actions)

            wp.launch(kernel=applyActionsKernel,
                      dim=len(self.laparoscopeXForm),
                      inputs=[self.laparoscopeXForm,
                              actions,
                              self.laparoscopeDragCutHeat,
                              self.laparoscopeDragCutHeatUpdate],
                      device=self.device)

            wp.launch(kernel=updateLaparoscopeKernel,
                      dim=4 * self.numEnvs,
                      inputs=[self.laparoscopeXForm,
                              self.laparoscopeTip,
                              self.laparoscopeBase,
                              self.laparoscopeRadius,
                              self.laparoscopeHeight],
                      device=self.device)

        return grabbed

    def forceLaparoscopeClampVertex(self, vertexId:int, envs, on:float=1.0, animate:bool=True):
        envIds = torch.nonzero(envs).to(dtype=torch.int32).flatten()
        envIds = wp.from_torch(envIds)

        wp.launch(kernel=forceClampsDragKernel, 
                  dim=len(envIds), 
                  inputs=[self.activeDragConstraint,
                          vertexId,
                          envIds,
                          on,
                          self.environmentNumVertices],
                  device=self.device)

        if animate:
            action = torch.tensor([0., 0., 0., 0., -1.], dtype=torch.float, device=self.device)
            actions = action.repeat(self.numEnvs, 1) * torch.transpose(envs.repeat(5, 1), 0, 1)
            actions = actions.flatten()

            actions = wp.from_torch(actions)
            
            wp.launch(kernel=applyActionsKernel,
                      dim=len(self.laparoscopeXForm),
                      inputs=[self.laparoscopeXForm,
                              actions,
                              self.laparoscopeDragCutHeat,
                              self.laparoscopeDragCutHeatUpdate],
                      device=self.device)

            wp.launch(kernel=updateLaparoscopeKernel, 
                      dim=4*self.numEnvs, 
                      inputs=[self.laparoscopeXForm,
                              self.laparoscopeTip,
                              self.laparoscopeBase,
                              self.laparoscopeRadius,
                              self.laparoscopeHeight],
                      device=self.device)
    
    def setRotationCenter(self, laparoscopeTranslation, laparoscopeRotation):

        wp.launch(kernel=setRotationCenterTransformKernel,
                  dim=self.numEnvs,
                  inputs=[self.laparoscopeXForm,
                          laparoscopeTranslation,
                          laparoscopeRotation],
                  device=self.device)
        
        wp.launch(kernel=updateLaparoscopeKernel, 
                  dim=4*self.numEnvs, 
                  inputs=[self.laparoscopeXForm,
                          self.laparoscopeTip,
                          self.laparoscopeBase,
                          self.laparoscopeRadius,
                          self.laparoscopeHeight],
                  device=self.device)

    # Can be done on GPU, but is it worth it?
    def resetFixedModel(self, envNums):
        zeroVertex = wp.zeros_like(self.vertex)
        zeroDragCutHeat = wp.zeros_like(self.laparoscopeDragCutHeat)

        for i in envNums:
            vertexOffset = i * self.numEnvAllVertices
            dragCutHeatOffset = i * 3
            # reset meshes
            wp.copy(self.vertex, self.initialVertex, dest_offset=vertexOffset, src_offset=vertexOffset, count=self.numEnvAllVertices)
            wp.copy(self.predictedVertex, zeroVertex, dest_offset=vertexOffset, src_offset=vertexOffset, count=self.numEnvAllVertices)
            wp.copy(self.velocity, zeroVertex, dest_offset=vertexOffset, src_offset=vertexOffset, count=self.numEnvAllVertices)

            laparoscopeXFormOffset = i * 7
            laparoscopeBaseTipOffset = i * 4
            # reset laparoscopes
            wp.copy(self.laparoscopeXForm, self.laparoscopeInitialXForm, dest_offset=laparoscopeXFormOffset, src_offset=laparoscopeXFormOffset, count=7)
            wp.copy(self.laparoscopeBase, self.laparoscopeInitialBase, dest_offset=laparoscopeBaseTipOffset, src_offset=laparoscopeBaseTipOffset, count=4)
            wp.copy(self.laparoscopeTip, self.laparoscopeInitialTip, dest_offset=laparoscopeBaseTipOffset, src_offset=laparoscopeBaseTipOffset, count=4)

            vertexOffset = i * self.numEnvAllVertices
            wp.copy(self.laparoscopeVertex, self.laparoscopeInitialVertex)

            wp.copy(self.laparoscopeDragCutHeat, zeroDragCutHeat, dest_offset=dragCutHeatOffset, src_offset=dragCutHeatOffset, count=3)
            wp.copy(self.laparoscopeDragCutHeatUpdate, zeroDragCutHeat, dest_offset=dragCutHeatOffset, src_offset=dragCutHeatOffset, count=3)

            zeroActiveVertex = wp.zeros_like(self.activeDragConstraint)
            wp.copy(self.activeDragConstraint, zeroActiveVertex, dest_offset=vertexOffset, src_offset=vertexOffset, count=self.numEnvAllVertices)

            laparoscopeVertexOffset = i * self.numEnvLaparoscopeVisPoints
            wp.copy(self.allVisPoint, self.allVisPointInitial, dest_offset=laparoscopeVertexOffset, src_offset=laparoscopeVertexOffset, count=self.numEnvLaparoscopeVisPoints)

            # Reset adhesion bonds for this environment
            if self.numAdhesionBonds > 0 and self.numAdhesionBondsPerEnv > 0:
                adhesionOffset = i * self.numAdhesionBondsPerEnv
                wp.copy(self.adhesionActive, self.adhesionInitialActive,
                        dest_offset=adhesionOffset, src_offset=adhesionOffset,
                        count=self.numAdhesionBondsPerEnv)

            # Reset rigid-deform adhesion bonds for this environment
            if self.numRigidAdhesionBonds > 0 and self.numRigidAdhesionBondsPerEnv > 0:
                rigidAdhOffset = i * self.numRigidAdhesionBondsPerEnv
                wp.copy(self.rigidAdhesionActive, self.rigidAdhesionInitialActive,
                        dest_offset=rigidAdhOffset, src_offset=rigidAdhOffset,
                        count=self.numRigidAdhesionBondsPerEnv)

        # Needs to run, otherwise laparoscope parts are overlapped in 0, 0, 0 coords. and BVH is problematic
        self.transformSimLaparoscopeMeshData()
        self.transformSimMeshData()

    def getAdhesionActiveTensor(self):
        """Get adhesion active states as PyTorch tensor, shape (numEnvs, bondsPerEnv)."""
        if self.numAdhesionBonds == 0:
            return None
        import torch
        activeTensor = wp.to_torch(self.adhesionActive)
        return activeTensor.view(self.numEnvs, self.numAdhesionBondsPerEnv)

    def getAdhesionActiveCounts(self):
        """Get count of active bonds per environment as PyTorch tensor, shape (numEnvs,)."""
        if self.numAdhesionBonds == 0:
            return None
        import torch
        activeTensor = wp.to_torch(self.adhesionActive)
        return activeTensor.view(self.numEnvs, self.numAdhesionBondsPerEnv).sum(dim=1)

    def createAdhesionBondsProgrammatic(self, vertexIds, triIds, triBar=None, restGap=None):
        """
        Create adhesion bonds programmatically (without USD).

        Args:
            vertexIds: list of vertex indices (per env, will be replicated for all envs)
            triIds: list of triangle vertex indices (3 per bond, flat, per env)
            triBar: optional list of barycentric coords (wp.vec3 per bond).
                    If None, computed from vertex positions via initAdhesionBondsKernel.
            restGap: optional list of rest distances (float per bond).
                     If None, computed from vertex positions.

        Bond arrays are replicated across all environments with proper index offsets.
        """
        from FF_SRL.adhesion import initAdhesionBondsKernel

        numBondsPerEnv = len(vertexIds)

        # Replicate across environments with vertex offset
        allVertexIds = []
        allTriIds = []
        for i in range(self.numEnvs):
            offset = i * self.numEnvAllVertices
            allVertexIds += [v + offset for v in vertexIds]
            allTriIds += [t + offset for t in triIds]

        totalBonds = numBondsPerEnv * self.numEnvs

        self.numAdhesionBonds = totalBonds
        self.numAdhesionBondsPerEnv = numBondsPerEnv

        self.adhesionVertexId = wp.array(allVertexIds, dtype=wp.int32, device=self.device)
        self.adhesionTriId = wp.array(allTriIds, dtype=wp.int32, device=self.device)

        if triBar is not None and restGap is not None:
            # Use provided values, replicate across envs
            allTriBar = triBar * self.numEnvs
            allRestGap = restGap * self.numEnvs
            self.adhesionTriBar = wp.array(allTriBar, dtype=wp.vec3, device=self.device)
            self.adhesionRestGap = wp.array(allRestGap, dtype=wp.float32, device=self.device)
        else:
            # Compute from vertex positions
            self.adhesionTriBar = wp.zeros(totalBonds, dtype=wp.vec3, device=self.device)
            self.adhesionRestGap = wp.zeros(totalBonds, dtype=wp.float32, device=self.device)

            wp.launch(kernel=initAdhesionBondsKernel,
                      dim=totalBonds,
                      inputs=[self.vertex,
                              self.adhesionVertexId,
                              self.adhesionTriId,
                              self.adhesionTriBar,
                              self.adhesionRestGap],
                      device=self.device)

        self.adhesionActive = wp.array([1.0] * totalBonds, dtype=wp.float32, device=self.device)
        self.adhesionInitialActive = wp.array([1.0] * totalBonds, dtype=wp.float32, device=self.device)
        self.adhesionNormalCache = wp.zeros(totalBonds, dtype=wp.vec3, device=self.device)
        self.adhesionCacheValid = wp.zeros(totalBonds, dtype=wp.int32, device=self.device)

        print(f"Created {totalBonds} adhesion bonds ({numBondsPerEnv} per env, {self.numEnvs} envs)")

    def createRigidAdhesionBondsProgrammatic(self, rigidPoints, triIds, triBar=None, restGap=None):
        """
        Create rigid-deform adhesion bonds programmatically.

        Args:
            rigidPoints: list of wp.vec3 - fixed world-space points on rigid body (1 per bond, per env)
            triIds: list of triangle vertex indices on deformable mesh (3 per bond, flat, per env)
            triBar: optional list of barycentric coords (wp.vec3 per bond).
                    If None, computed from vertex positions via initRigidAdhesionBondsKernel.
            restGap: optional list of rest distances (float per bond).
                     If None, computed from vertex positions.

        Bond arrays are replicated across all environments with proper index offsets.
        """
        from FF_SRL.adhesion import initRigidAdhesionBondsKernel

        numBondsPerEnv = len(rigidPoints)

        allRigidPoints = []
        allTriIds = []
        for i in range(self.numEnvs):
            offset = i * self.numEnvAllVertices
            allRigidPoints += rigidPoints  # rigid points don't need offset (world-space, same per env)
            allTriIds += [x + offset for x in triIds]

        totalBonds = numBondsPerEnv * self.numEnvs

        self.numRigidAdhesionBonds = totalBonds
        self.numRigidAdhesionBondsPerEnv = numBondsPerEnv

        self.rigidAdhesionRigidPoint = wp.array(allRigidPoints, dtype=wp.vec3, device=self.device)
        self.rigidAdhesionTriId = wp.array(allTriIds, dtype=wp.int32, device=self.device)

        if triBar is not None and restGap is not None:
            allTriBar = triBar * self.numEnvs
            allRestGap = restGap * self.numEnvs
            self.rigidAdhesionTriBar = wp.array(allTriBar, dtype=wp.vec3, device=self.device)
            self.rigidAdhesionRestGap = wp.array(allRestGap, dtype=wp.float32, device=self.device)
        else:
            self.rigidAdhesionTriBar = wp.zeros(totalBonds, dtype=wp.vec3, device=self.device)
            self.rigidAdhesionRestGap = wp.zeros(totalBonds, dtype=wp.float32, device=self.device)

            wp.launch(kernel=initRigidAdhesionBondsKernel,
                      dim=totalBonds,
                      inputs=[self.vertex,
                              self.rigidAdhesionRigidPoint,
                              self.rigidAdhesionTriId,
                              self.rigidAdhesionTriBar,
                              self.rigidAdhesionRestGap],
                      device=self.device)

        self.rigidAdhesionActive = wp.array([1.0] * totalBonds, dtype=wp.float32, device=self.device)
        self.rigidAdhesionInitialActive = wp.array([1.0] * totalBonds, dtype=wp.float32, device=self.device)
        self.rigidAdhesionNormalCache = wp.zeros(totalBonds, dtype=wp.vec3, device=self.device)
        self.rigidAdhesionCacheValid = wp.zeros(totalBonds, dtype=wp.int32, device=self.device)

        print(f"Created {totalBonds} rigid-deform adhesion bonds ({numBondsPerEnv} per env, {self.numEnvs} envs)")

    def createRigidAdhesionFromMeshes(self, bondDistance=3.5):
        """
        Auto-create rigid-deform adhesion bonds between rigid body (bone) surface
        and deformable mesh (tumor) surface triangles.

        Algorithm (matching xpbd-tissue-sim Simulation.cpp):
          1. Get bone surface vertices (rigid body)
          2. For each tumor surface triangle, compute centroid
          3. Find closest bone vertex to each centroid
          4. If distance < bondDistance, create a bond:
             - rigidPoint = closest bone surface vertex (world space, fixed)
             - triIds = tumor triangle vertex indices (deformable)

        Args:
            bondDistance: maximum distance (cm) for bond creation.
                         Default 3.5cm (= 35mm, matching xpbd-tissue-sim's 0.0348m)
        """
        import numpy as np
        from scipy.spatial import cKDTree

        # Get rigid body vertices (bone surface)
        rigidVerts = []
        for simRigid in self.simEnvironment.simRigids:
            rigidVerts.append(np.array(simRigid.vertex))
        if len(rigidVerts) == 0:
            print("No rigid bodies found, skipping rigid-deform adhesion creation")
            return
        rigidVerts = np.vstack(rigidVerts)

        # Get deformable mesh surface triangles (tumor)
        # Use first env's vertex positions and triangle indices
        simMesh = self.simEnvironment.simMeshes[0]
        tumorVerts = np.array(simMesh.vertex)
        tumorTriIndices = np.array(simMesh.triangle).reshape(-1, 3)  # surface tri indices

        print(f"  Rigid (bone) vertices: {len(rigidVerts)}")
        print(f"  Deformable (tumor) surface triangles: {len(tumorTriIndices)}")
        print(f"  Bond distance threshold: {bondDistance:.2f} cm")

        # Build KD-tree on bone vertices for fast nearest-neighbor lookup
        tree = cKDTree(rigidVerts)

        # For each tumor surface triangle, find closest bone point
        bondRigidPoints = []  # wp.vec3 list
        bondTriIds = []       # flat list of 3 vertex indices per bond

        for tri in tumorTriIndices:
            # Triangle centroid
            centroid = tumorVerts[tri].mean(axis=0)

            # Query closest bone vertex
            dist, idx = tree.query(centroid)

            if dist < bondDistance:
                bonePoint = rigidVerts[idx]
                bondRigidPoints.append(tuple(bonePoint))
                bondTriIds.extend(tri.tolist())

        numBonds = len(bondRigidPoints)
        if numBonds == 0:
            print(f"  WARNING: No bonds created! Increase bondDistance (current={bondDistance:.2f}cm)")
            print(f"  Tumor bbox: {tumorVerts.min(0)} -> {tumorVerts.max(0)}")
            print(f"  Bone bbox: {rigidVerts.min(0)} -> {rigidVerts.max(0)}")
            return

        print(f"  Created {numBonds} bonds out of {len(tumorTriIndices)} triangles")

        # Use existing programmatic API (handles multi-env replication + GPU init)
        self.createRigidAdhesionBondsProgrammatic(bondRigidPoints, bondTriIds)

    def getRigidAdhesionActiveTensor(self):
        """Get rigid adhesion active states as PyTorch tensor, shape (numEnvs, bondsPerEnv)."""
        if self.numRigidAdhesionBonds == 0:
            return None
        import torch
        activeTensor = wp.to_torch(self.rigidAdhesionActive)
        return activeTensor.view(self.numEnvs, self.numRigidAdhesionBondsPerEnv)

    def getRigidAdhesionActiveCounts(self):
        """Get count of active rigid-deform bonds per environment as PyTorch tensor, shape (numEnvs,)."""
        if self.numRigidAdhesionBonds == 0:
            return None
        import torch
        activeTensor = wp.to_torch(self.rigidAdhesionActive)
        return activeTensor.view(self.numEnvs, self.numRigidAdhesionBondsPerEnv).sum(dim=1)

    def resetStochasticModel(self, laparoscopeTranslation, laparoscopeRotation):
        # reset meshes
        wp.copy(self.vertex, self.initialVertex)

        # reset laparoscopes
        zeroDragCutHeat = wp.zeros_like(self.laparoscopeDragCutHeat)
        
        wp.copy(self.laparoscopeXForm, self.laparoscopeInitialXForm)
        wp.copy(self.laparoscopeBase, self.laparoscopeInitialBase)
        wp.copy(self.laparoscopeTip, self.laparoscopeInitialTip)

        wp.copy(self.laparoscopeVertex, self.laparoscopeInitialVertex)

        wp.copy(self.laparoscopeDragCutHeat, zeroDragCutHeat)
        wp.copy(self.laparoscopeDragCutHeatUpdate, zeroDragCutHeat)

        wp.copy(self.allVisPoint, self.allVisPointInitial)

        self.setRotationCenter(laparoscopeTranslation, laparoscopeRotation)
        
        self.transformSimLaparoscopeMeshData()
        self.transformSimMeshData()

class SimEnvironmentDO():

    def __init__(self, stage, device, envNumber, dbInputs=None,
                 globalLaparoscopeDragLookupRadius=0.15):
        
        self.simMeshes = []
        self.simLockBoxes = []
        self.simConnectors = []
        self.simAdhesions = []
        self.simLaparoscopes = []
        self.simRigids = []
        self.envNumber = envNumber
        self.stage = stage

        self.globalLaparoscopeDragLookupRadius = globalLaparoscopeDragLookupRadius
        self.environmentNumVertices = 0

        meshes = [x for x in self.stage.Traverse() if x.IsA(UsdGeom.Mesh) and x.GetAttribute("simMesh").Get() == True]
        lockBoxes = [x for x in self.stage.Traverse() if x.IsA(UsdGeom.Cube) and x.GetAttribute("simLockBox").Get() == True]
        connectors = [x for x in self.stage.Traverse() if x.IsA(UsdGeom.Xform) and x.GetAttribute("simConnector").Get() == True]
        adhesions = [x for x in self.stage.Traverse() if x.IsA(UsdGeom.Xform) and x.GetAttribute("simAdhesion").Get() == True]
        laparoscopes = [x for x in self.stage.Traverse() if x.IsA(UsdGeom.Xform) and x.GetAttribute("simLaparoscope").Get() == True]
        rigids = [x for x in self.stage.Traverse() if x.IsA(UsdGeom.Mesh) and x.GetAttribute("simRigid").Get() == True]

        for mesh in meshes:
            simMesh = dk.SimMeshDO(mesh, device)
            self.simMeshes.append(simMesh)
            self.environmentNumVertices += simMesh.numVertices

        for lockBox in lockBoxes:
            simLockBox = dk.SimLockBoxDO(lockBox, device)
            self.simLockBoxes.append(simLockBox)

        for connector in connectors:
            simConnector = dk.SimConnectorDO(connector, device)
            self.simConnectors.append(simConnector)

        for adhesion in adhesions:
            simAdhesion = SimAdhesionDO(adhesion, device)
            self.simAdhesions.append(simAdhesion)

        for laparoscope in laparoscopes:
            simLaparoscope = dk.SimLaparoscopeDO(laparoscope, device, self.stage, laparoscopeDragLookupRadius = self.globalLaparoscopeDragLookupRadius)
            self.simLaparoscopes.append(simLaparoscope)

        for rigid in rigids:
            simRigid = dk.SimRigidDO(rigid, device)
            self.simRigids.append(simRigid)

        for simMesh in self.simMeshes:

            for simLockBox in self.simLockBoxes:

                simLockBox.lockArray(simMesh.vertex, simMesh.inverseMass)

    def duplicateMeshes(self, spacingX:float, spacingZ:float):

        xFormShift = ((self.envNumber//8)*spacingX, 0.0, (self.envNumber%8)*spacingZ)

        for i in range(len(self.simMeshes)):
            simMesh = self.simMeshes[i]
            simMesh.name = simMesh.name + str(self.envNumber)
            simMesh.path = Sdf.Path("/World/" + simMesh.name)

            meshGeom = UsdGeom.Mesh.Define(self.stage, simMesh.path)
            meshGeom.CreatePointsAttr(simMesh.meshVisPoints)
            meshGeom.CreateFaceVertexCountsAttr([3]*simMesh.numTris)
            meshGeom.CreateFaceVertexIndicesAttr(simMesh.meshVisFaces)
            meshGeom.CreateNormalsAttr(simMesh.meshNormals)
            meshGeom.SetNormalsInterpolation("vertex")
            meshGeom.CreateSubdivisionSchemeAttr().Set("none")
            
            simMesh.xForm = UsdGeom.Xform(meshGeom)
            t = simMesh.xForm.AddTranslateOp()
            r = simMesh.xForm.AddOrientOp()
            s = simMesh.xForm.AddScaleOp()
            t.Set(simMesh.translation + xFormShift)
            # ToDo: DAMN! XYZ
            r.Set(Gf.Quatf((Gf.Rotation(Gf.Vec3d(0, 0, 1), simMesh.rotation[2])*
                            Gf.Rotation(Gf.Vec3d(0, 1, 0), simMesh.rotation[1])*
                            Gf.Rotation(Gf.Vec3d(1, 0, 0), simMesh.rotation[0])).GetQuat()))
            s.Set(simMesh.scale)

            simMesh.prim = self.stage.GetPrimAtPath(simMesh.path)

    def duplicateLaparoscopes(self, spacingX:float, spacingZ:float):
        
        xFormShift = ((self.envNumber//8)*spacingX, 0.0, (self.envNumber%8)*spacingZ)

        # ToDo: Move to cloning if possible
        for simLaparoscope in self.simLaparoscopes:

            simLaparoscope.name = "simpleLaparoscope" + str(self.envNumber)
            simLaparoscope.path = Sdf.Path("/World/" + simLaparoscope.name)
            simLaparoscope.xForm = UsdGeom.Xform.Define(self.stage, simLaparoscope.path)
            simLaparoscope.prim = self.stage.GetPrimAtPath(simLaparoscope.path)

            simLaparoscope.xForm.ClearXformOpOrder()
            t = simLaparoscope.xForm.AddTranslateOp()
            r = simLaparoscope.xForm.AddOrientOp()
            s = simLaparoscope.xForm.AddScaleOp()
            t.Set(simLaparoscope.translation + xFormShift)
            r.Set(Gf.Quatf((Gf.Rotation(Gf.Vec3d(1, 0, 0), simLaparoscope.rotation[0])*
                            Gf.Rotation(Gf.Vec3d(0, 1, 0), simLaparoscope.rotation[1])*
                            Gf.Rotation(Gf.Vec3d(0, 0, 1), simLaparoscope.rotation[2])).GetQuat()))

            simLaparoscope.prim = self.stage.GetPrimAtPath(simLaparoscope.path)

            simLaparoscope.rotationCenterPath = Sdf.Path(simLaparoscope.path.pathString + "/rotationCenter")
            simLaparoscope.rotationCenterXForm = UsdGeom.Sphere.Define(self.stage, simLaparoscope.rotationCenterPath)
            simLaparoscope.rotationCenterXForm.CreateRadiusAttr(simLaparoscope.rodRadius * 1.5)
            simLaparoscope.rotationCenterXForm.AddTranslateOp()
            simLaparoscope.rotationCenterXForm.AddOrientOp()
            simLaparoscope.rotationCenterXForm.AddScaleOp()
            simLaparoscope.rotationCenterPrim = self.stage.GetPrimAtPath(simLaparoscope.rotationCenterPath)

            simLaparoscope.rodPath = Sdf.Path(simLaparoscope.rotationCenterPath.pathString + "/rod")
            simLaparoscope.rodXForm = UsdGeom.Capsule.Define(self.stage, simLaparoscope.rodPath)
            simLaparoscope.rodXForm.CreateRadiusAttr(simLaparoscope.rodRadius)
            simLaparoscope.rodXForm.CreateHeightAttr(simLaparoscope.rodHeight)
            simLaparoscope.rodXForm.AddTranslateOp()
            simLaparoscope.rodXForm.AddOrientOp()
            simLaparoscope.rodXForm.AddScaleOp()
            simLaparoscope.rodPrim = self.stage.GetPrimAtPath(simLaparoscope.rodPath)

            clampHeightScale = 1.0/5.0
            clampRadiusScale = 3.0/5.0

            simLaparoscope.leftClampRotationCenterPath = Sdf.Path(simLaparoscope.rodPath.pathString + "/leftClampRotationCenter")
            simLaparoscope.leftClampRotationCenterXForm = UsdGeom.Xform.Define(self.stage, simLaparoscope.leftClampRotationCenterPath)
            t = simLaparoscope.leftClampRotationCenterXForm.AddTranslateOp()
            r = simLaparoscope.leftClampRotationCenterXForm.AddOrientOp()
            s = simLaparoscope.leftClampRotationCenterXForm.AddScaleOp()
            t.Set((0.0, 0.0, simLaparoscope.rodHeight * 0.5))
            simLaparoscope.leftClampRotationCenterPrim = self.stage.GetPrimAtPath(simLaparoscope.leftClampRotationCenterPath)

            simLaparoscope.leftClampPath = Sdf.Path(simLaparoscope.leftClampRotationCenterPath.pathString + "/leftClamp")
            simLaparoscope.leftClampXForm = UsdGeom.Capsule.Define(self.stage, simLaparoscope.leftClampPath)
            simLaparoscope.leftClampXForm.CreateRadiusAttr(simLaparoscope.rodRadius * clampRadiusScale)
            simLaparoscope.leftClampXForm.CreateHeightAttr(simLaparoscope.rodHeight * clampHeightScale)
            t = simLaparoscope.leftClampXForm.AddTranslateOp()
            r = simLaparoscope.leftClampXForm.AddOrientOp()
            s = simLaparoscope.leftClampXForm.AddScaleOp()
            t.Set((0.0, simLaparoscope.rodRadius, simLaparoscope.rodHeight * clampHeightScale * 0.4))
            r.Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), -30.0).GetQuat()))
            simLaparoscope.leftClampPrim = self.stage.GetPrimAtPath(simLaparoscope.leftClampPath)

            simLaparoscope.rightClampRotationCenterPath = Sdf.Path(simLaparoscope.rodPath.pathString + "/rightClampRotationCenter")
            simLaparoscope.rightClampRotationCenterXForm = UsdGeom.Xform.Define(self.stage, simLaparoscope.rightClampRotationCenterPath)
            t = simLaparoscope.rightClampRotationCenterXForm.AddTranslateOp()
            r = simLaparoscope.rightClampRotationCenterXForm.AddOrientOp()
            s = simLaparoscope.rightClampRotationCenterXForm.AddScaleOp()
            t.Set((0.0, 0.0, simLaparoscope.rodHeight * 0.5))
            simLaparoscope.rightClampRotationCenterPrim = self.stage.GetPrimAtPath(simLaparoscope.rightClampRotationCenterPath)

            simLaparoscope.rightClampPath = Sdf.Path(simLaparoscope.rightClampRotationCenterPath.pathString + "/rightClamp")
            simLaparoscope.rightClampXForm = UsdGeom.Capsule.Define(self.stage, simLaparoscope.rightClampPath)
            simLaparoscope.rightClampXForm.CreateRadiusAttr(simLaparoscope.rodRadius * clampRadiusScale)
            simLaparoscope.rightClampXForm.CreateHeightAttr(simLaparoscope.rodHeight * clampHeightScale)
            t = simLaparoscope.rightClampXForm.AddTranslateOp()
            r = simLaparoscope.rightClampXForm.AddOrientOp()
            s = simLaparoscope.rightClampXForm.AddScaleOp()
            t.Set((0.0, -simLaparoscope.rodRadius, simLaparoscope.rodHeight * clampHeightScale * 0.4))
            r.Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 30.0).GetQuat()))
            simLaparoscope.rightClampPrim = self.stage.GetPrimAtPath(simLaparoscope.rightClampPath)

    def removeOriginalMeshes(self):
        for simMesh in self.simMeshes:
            self.stage.RemovePrim(simMesh.path)

        # xForms = [x for x in self.stage.Traverse() if x.IsA(UsdGeom.Xform)]
        # for xForm in xForms:
        #     self.stage.RemovePrim(xForm)

    def removeOriginalLaparoscopes(self):
        for simLaparoscope in self.simLaparoscopes:
            self.stage.RemovePrim(simLaparoscope.path)
