#include "AttitudeEngine.h"
#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <cmath>

AttitudeEngine::AttitudeEngine(QObject *parent)
    : QObject(parent)
    , simulationTimer(new QTimer(this))
    , currentSimTime(QDateTime::currentDateTime())
    , simSpeed(1.0)
    , isRunning(false)
    , attitude(Eigen::Quaterniond::Identity())
    , bodyRates(Eigen::Vector3d::Zero())
    , sunDirection(Eigen::Vector3d(1.0, 0.0, 0.0))
    , nadirDirection(Eigen::Vector3d(0.0, 0.0, -1.0))
    , velocityVector(Eigen::Vector3d(0.0, 1.0, 0.0))
    , referenceFrame("ECI")
{
    // Connect the timer to update function
    connect(simulationTimer, &QTimer::timeout, this, &AttitudeEngine::updateSimulation);
    
    // Set timer interval (milliseconds)
    simulationTimer->setInterval(16); // 60 fps
    
    // Initialize enabled vectors
    enabledVectors[BODY_X] = true;
    enabledVectors[BODY_Y] = true;
    enabledVectors[BODY_Z] = true;
    enabledVectors[NADIR] = true;
    enabledVectors[SUN] = true;
    enabledVectors[VELOCITY] = false;
    enabledVectors[ANGULAR_MOMENTUM] = false;
}

AttitudeEngine::~AttitudeEngine()
{
    if (simulationTimer->isActive()) {
        simulationTimer->stop();
    }
}

void AttitudeEngine::startSimulation()
{
    if (!isRunning) {
        isRunning = true;
        simulationTimer->start();
        qDebug() << "Simulation started";
    }
}

void AttitudeEngine::stopSimulation()
{
    if (isRunning) {
        isRunning = false;
        simulationTimer->stop();
        qDebug() << "Simulation stopped";
    }
}

void AttitudeEngine::setSimulationSpeed(double speed)
{
    simSpeed = speed;
    qDebug() << "Simulation speed set to" << speed;
}

void AttitudeEngine::setSimulationDate(const QDateTime &dateTime)
{
    currentSimTime = dateTime;
    emit simulationTimeChanged(currentSimTime.toString("yyyy-MM-dd hh:mm:ss"));
    qDebug() << "Simulation date set to" << dateTime;
    
    // Update sun direction based on new date
    sunDirection = calculateSunDirection();
    
    // Update vectors and emit signals
    updateVectors();
}

void AttitudeEngine::loadMockData(const QString &dataPath)
{
    QFile file(dataPath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Could not open mock data file:" << dataPath;
        return;
    }
    
    QByteArray data = file.readAll();
    file.close();
    
    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull() || !doc.isObject()) {
        qWarning() << "Invalid JSON data in mock file";
        return;
    }
    
    QJsonObject obj = doc.object();
    
    // Example of parsing mock attitude data
    if (obj.contains("attitude") && obj["attitude"].isObject()) {
        QJsonObject attObj = obj["attitude"].toObject();
        double w = attObj["w"].toDouble();
        double x = attObj["x"].toDouble();
        double y = attObj["y"].toDouble();
        double z = attObj["z"].toDouble();
        
        // Set the quaternion
        Eigen::Quaterniond quat(w, x, y, z);
        quat.normalize();
        setQuaternion(quat);
        
        qDebug() << "Loaded mock attitude quaternion:" << w << x << y << z;
    }
    
    // Example of parsing mock body rates
    if (obj.contains("bodyRates") && obj["bodyRates"].isObject()) {
        QJsonObject ratesObj = obj["bodyRates"].toObject();
        double wx = ratesObj["x"].toDouble();
        double wy = ratesObj["y"].toDouble();
        double wz = ratesObj["z"].toDouble();
        
        // Set the body rates
        setBodyRates(wx, wy, wz);
        
        qDebug() << "Loaded mock body rates:" << wx << wy << wz;
    }
}

void AttitudeEngine::setQuaternion(const Eigen::Quaterniond &quat)
{
    attitude = quat.normalized();
    emit attitudeUpdated(attitude);
    
    // Update vectors based on new attitude
    updateVectors();
    
    qDebug() << "Set attitude quaternion w:" << attitude.w() 
             << "x:" << attitude.x() 
             << "y:" << attitude.y() 
             << "z:" << attitude.z();
}

void AttitudeEngine::setBodyRates(double wx, double wy, double wz)
{
    bodyRates = Eigen::Vector3d(wx, wy, wz);
    emit bodyRatesUpdated(wx, wy, wz);
    
    qDebug() << "Set body rates:" << wx << wy << wz;
}

void AttitudeEngine::setReferenceFrame(const QString &frameName)
{
    referenceFrame = frameName;
    qDebug() << "Reference frame set to:" << frameName;
    
    // Update vectors for the new reference frame
    updateVectors();
}

void AttitudeEngine::toggleVector(VectorType vectorType, bool enabled)
{
    enabledVectors[vectorType] = enabled;
    
    qDebug() << "Vector" << vectorType << (enabled ? "enabled" : "disabled");
    
    // Update the vector if it's now enabled
    if (enabled) {
        updateVectors();
    }
}

void AttitudeEngine::updateSimulation()
{
    // Use exponential scaling for the simulation speed
    double scaledSpeed = std::pow(simSpeed, 2.5); // Exponential scaling
    
    // Advance time by larger increments for higher speeds
    int timeAdvanceSeconds = static_cast<int>(5.0 * scaledSpeed);
    timeAdvanceSeconds = std::max(1, timeAdvanceSeconds); // At least 1 second
    
    currentSimTime = currentSimTime.addSecs(timeAdvanceSeconds);
    
    // Emit the updated time
    emit simulationTimeChanged(currentSimTime.toString("yyyy-MM-dd hh:mm:ss"));
    
    // For a simple simulation, update the attitude based on body rates
    // This is a simple Euler integration of q_dot = 0.5 * q * omega
    if (bodyRates.norm() > 1e-10) {
        double dt = 0.016 * scaledSpeed;  // Timestep in seconds (scaled)
        
        // Create a quaternion from body rates (angular velocity vector)
        Eigen::Vector3d rotation = bodyRates * dt;
        double angle = rotation.norm();
        
        if (angle > 1e-10) {
            Eigen::Quaterniond deltaQ;
            deltaQ = Eigen::AngleAxisd(angle, rotation.normalized());
            
            // Update the attitude quaternion
            attitude = (attitude * deltaQ).normalized();
            
            // Emit the updated attitude
            emit attitudeUpdated(attitude);
            
            // Update vectors based on new attitude
            updateVectors();
        }
    }
    
    // Update sun direction based on time
    sunDirection = calculateSunDirection();
    
    // Update other reference vectors
    nadirDirection = calculateNadirDirection();
    velocityVector = calculateVelocityVector();
    
    // Update all vectors
    updateVectors();
}

void AttitudeEngine::updateVectors()
{
    // Update body axes vectors in the reference frame
    if (enabledVectors[BODY_X]) {
        Eigen::Vector3d bodyX = attitude * Eigen::Vector3d::UnitX();
        emit vectorUpdated(BODY_X, bodyX);
    }
    
    if (enabledVectors[BODY_Y]) {
        Eigen::Vector3d bodyY = attitude * Eigen::Vector3d::UnitY();
        emit vectorUpdated(BODY_Y, bodyY);
    }
    
    if (enabledVectors[BODY_Z]) {
        Eigen::Vector3d bodyZ = attitude * Eigen::Vector3d::UnitZ();
        emit vectorUpdated(BODY_Z, bodyZ);
    }
    
    // Update reference vectors in body frame
    if (enabledVectors[NADIR]) {
        // Transform nadir direction to body frame
        Eigen::Vector3d nadirInBody = attitude.inverse() * nadirDirection;
        emit vectorUpdated(NADIR, nadirInBody);
    }
    
    if (enabledVectors[SUN]) {
        // Transform sun direction to body frame
        Eigen::Vector3d sunInBody = attitude.inverse() * sunDirection;
        emit vectorUpdated(SUN, sunInBody);
    }
    
    if (enabledVectors[VELOCITY]) {
        // Transform velocity vector to body frame
        Eigen::Vector3d velocityInBody = attitude.inverse() * velocityVector;
        emit vectorUpdated(VELOCITY, velocityInBody);
    }
    
    if (enabledVectors[ANGULAR_MOMENTUM]) {
        // Angular momentum is proportional to body rates for a simple model
        // H = I * omega, but we'll use a simplified version here
        Eigen::Vector3d angMomentum = bodyRates.normalized();
        emit vectorUpdated(ANGULAR_MOMENTUM, angMomentum);
    }
}

Eigen::Vector3d AttitudeEngine::calculateSunDirection()
{
    // Simple model for sun direction based on time of year
    // This is a very simplified model for visualization purposes
    
    // Get day of year (0-365)
    QDate date = currentSimTime.date();
    int dayOfYear = date.dayOfYear() - 1;  // 0-based
    
    // Calculate sun position using a simplified model
    // Earth's axial tilt is about 23.44 degrees
    double axialTilt = 23.44 * M_PI / 180.0;
    
    // Angle of Earth in its orbit around the sun (simplified)
    double orbitalAngle = 2.0 * M_PI * dayOfYear / 365.0;
    
    // Approximate sun direction in ECI coordinates
    double x = cos(orbitalAngle);
    double y = sin(orbitalAngle) * cos(axialTilt);
    double z = sin(orbitalAngle) * sin(axialTilt);
    
    return Eigen::Vector3d(x, y, z).normalized();
}

Eigen::Vector3d AttitudeEngine::calculateNadirDirection()
{
    // Simple model for nadir direction based on a circular orbit
    // This is a simplified model for visualization purposes
    
    // For now, just use a simple circular orbit model
    // In a real implementation, this would be based on orbital elements
    
    // Get time of day to vary the nadir direction
    QTime time = currentSimTime.time();
    double hourOfDay = time.hour() + time.minute()/60.0 + time.second()/3600.0;
    double angle = 2.0 * M_PI * hourOfDay / 24.0;
    
    // Calculate nadir direction (pointing from satellite to Earth center)
    double x = -cos(angle);
    double y = -sin(angle);
    double z = 0.0;  // Assuming equatorial orbit for simplicity
    
    return Eigen::Vector3d(x, y, z).normalized();
}

Eigen::Vector3d AttitudeEngine::calculateVelocityVector()
{
    // Simple model for velocity vector based on a circular orbit
    // This is a simplified model for visualization purposes
    
    // Velocity is perpendicular to nadir direction in a circular orbit
    Eigen::Vector3d nadir = calculateNadirDirection();
    
    // Calculate velocity direction (perpendicular to nadir in orbital plane)
    double x = -nadir.y();
    double y = nadir.x();
    double z = 0.0;  // Assuming equatorial orbit for simplicity
    
    return Eigen::Vector3d(x, y, z).normalized();
}