#include "MainWindow.h"
#include "ui_mainwindow.h"
#include "visualization/SpacecraftView.h"
#include "attitude/AttitudeEngine.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QGroupBox>
#include <QLabel>
#include <QSlider>
#include <QDateTimeEdit>
#include <QPushButton>
#include <QComboBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QColorDialog>
#include <QTabWidget>
#include <QFormLayout>
#include <QMessageBox>
#include <QDebug>
#include <QValidator>
#include <cmath>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , spacecraftView(nullptr)
    , attitudeEngine(nullptr)
{
    ui->setupUi(this);
    
    // Create the attitude engine
    attitudeEngine = new AttitudeEngine(this);
    
    // Setup UI components
    setupUI();
    
    // Create menu actions
    createActions();
    
    // Setup signal/slot connections
    setupConnections();
    
    // Set window title
    setWindowTitle("Spacecraft Attitude Visualization Tool");
    
    // Set status message
    statusBar()->showMessage("Ready", 3000);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setupUI()
{
    // Create spacecraft view
    spacecraftView = new SpacecraftView(ui->centralwidget);
    spacecraftView->setAttitudeEngine(attitudeEngine);
    
    // Create control panel
    QWidget *controlPanel = new QWidget(ui->centralwidget);
    QVBoxLayout *controlLayout = new QVBoxLayout(controlPanel);
    
    // Create tabbed interface for controls
    controlTabs = new QTabWidget(controlPanel);
    
    // ---- Simulation Tab ----
    QWidget *simTab = new QWidget();
    QVBoxLayout *simLayout = new QVBoxLayout(simTab);
    
    // Create a groupbox for simulation controls
    QGroupBox *simControlsGroup = new QGroupBox("Simulation Controls", simTab);
    QVBoxLayout *simControlsLayout = new QVBoxLayout(simControlsGroup);
    
    // Time display
    QLabel *timeLabel = new QLabel("Current Time:", simControlsGroup);
    timeDisplay = new QLabel("2023-01-01 00:00:00", simControlsGroup);
    timeDisplay->setObjectName("timeDisplay");
    timeDisplay->setFrameStyle(QFrame::Panel | QFrame::Sunken);
    timeDisplay->setAlignment(Qt::AlignCenter);
    
    // Start/Stop buttons
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    startButton = new QPushButton("Start", simControlsGroup);
    startButton->setObjectName("startButton");
    stopButton = new QPushButton("Stop", simControlsGroup);
    stopButton->setObjectName("stopButton");
    buttonLayout->addWidget(startButton);
    buttonLayout->addWidget(stopButton);
    
    // Simulation speed control
    QLabel *speedLabel = new QLabel("Simulation Speed:", simControlsGroup);
    speedSlider = new QSlider(Qt::Horizontal, simControlsGroup);
    speedSlider->setObjectName("speedSlider");
    speedSlider->setRange(1, 100);
    speedSlider->setValue(10);
    speedSlider->setTickPosition(QSlider::TicksBelow);
    speedSlider->setTickInterval(10);
    
    // Date/time selector
    QLabel *dateLabel = new QLabel("Set Date/Time:", simControlsGroup);
    dateTimeEdit = new QDateTimeEdit(QDateTime::currentDateTime(), simControlsGroup);
    dateTimeEdit->setObjectName("dateTimeEdit");
    dateTimeEdit->setCalendarPopup(true);
    dateTimeEdit->setDisplayFormat("yyyy-MM-dd hh:mm:ss");
    
    // Add widgets to simulation controls layout
    simControlsLayout->addWidget(timeLabel);
    simControlsLayout->addWidget(timeDisplay);
    simControlsLayout->addLayout(buttonLayout);
    simControlsLayout->addWidget(speedLabel);
    simControlsLayout->addWidget(speedSlider);
    simControlsLayout->addWidget(dateLabel);
    simControlsLayout->addWidget(dateTimeEdit);
    
    // Spacecraft model loader
    QPushButton *loadModelButton = new QPushButton("Load Spacecraft Model", simControlsGroup);
    simControlsLayout->addWidget(loadModelButton);
    
    // Load mock data button
    QPushButton *loadMockDataButton = new QPushButton("Load Mock Telemetry", simControlsGroup);
    simControlsLayout->addWidget(loadMockDataButton);
    
    // Reference frame selector
    QLabel *refFrameLabel = new QLabel("Reference Frame:", simControlsGroup);
    referenceFrameCombo = new QComboBox(simControlsGroup);
    referenceFrameCombo->addItem("ECI (Earth-Centered Inertial)");
    referenceFrameCombo->addItem("ECEF (Earth-Centered Earth-Fixed)");
    referenceFrameCombo->addItem("LVLH (Local Vertical/Local Horizontal)");
    simControlsLayout->addWidget(refFrameLabel);
    simControlsLayout->addWidget(referenceFrameCombo);
    
    simControlsLayout->addStretch();
    
    // Add the simulation controls to the tab
    simLayout->addWidget(simControlsGroup);
    
    // ---- Attitude Tab ----
    QWidget *attitudeTab = new QWidget();
    QVBoxLayout *attitudeLayout = new QVBoxLayout(attitudeTab);
    
    // Quaternion input
    QGroupBox *quatGroup = new QGroupBox("Quaternion Input", attitudeTab);
    QFormLayout *quatLayout = new QFormLayout(quatGroup);
    
    // Create quaternion input fields
    quatWInput = new QLineEdit(quatGroup);
    quatWInput->setText("1.0");
    quatWInput->setValidator(new QDoubleValidator(-1.0, 1.0, 6, this));
    
    quatXInput = new QLineEdit(quatGroup);
    quatXInput->setText("0.0");
    quatXInput->setValidator(new QDoubleValidator(-1.0, 1.0, 6, this));
    
    quatYInput = new QLineEdit(quatGroup);
    quatYInput->setText("0.0");
    quatYInput->setValidator(new QDoubleValidator(-1.0, 1.0, 6, this));
    
    quatZInput = new QLineEdit(quatGroup);
    quatZInput->setText("0.0");
    quatZInput->setValidator(new QDoubleValidator(-1.0, 1.0, 6, this));
    
    setQuatButton = new QPushButton("Set Quaternion", quatGroup);
    
    // Add to form layout
    quatLayout->addRow("W:", quatWInput);
    quatLayout->addRow("X:", quatXInput);
    quatLayout->addRow("Y:", quatYInput);
    quatLayout->addRow("Z:", quatZInput);
    quatLayout->addRow("", setQuatButton);
    
    // Euler angles input
    QGroupBox *eulerGroup = new QGroupBox("Euler Angles Input", attitudeTab);
    QFormLayout *eulerLayout = new QFormLayout(eulerGroup);
    
    // Create euler angle input fields
    rollInput = new QDoubleSpinBox(eulerGroup);
    rollInput->setRange(-180.0, 180.0);
    rollInput->setDecimals(2);
    rollInput->setValue(0.0);
    rollInput->setSuffix("°");
    
    pitchInput = new QDoubleSpinBox(eulerGroup);
    pitchInput->setRange(-90.0, 90.0);
    pitchInput->setDecimals(2);
    pitchInput->setValue(0.0);
    pitchInput->setSuffix("°");
    
    yawInput = new QDoubleSpinBox(eulerGroup);
    yawInput->setRange(-180.0, 180.0);
    yawInput->setDecimals(2);
    yawInput->setValue(0.0);
    yawInput->setSuffix("°");
    
    eulerConventionCombo = new QComboBox(eulerGroup);
    eulerConventionCombo->addItem("XYZ (Roll-Pitch-Yaw)");
    eulerConventionCombo->addItem("ZYX (Yaw-Pitch-Roll)");
    eulerConventionCombo->addItem("ZXZ (Aerospace)");
    
    setEulerButton = new QPushButton("Set Euler Angles", eulerGroup);
    
    // Add to form layout
    eulerLayout->addRow("Roll:", rollInput);
    eulerLayout->addRow("Pitch:", pitchInput);
    eulerLayout->addRow("Yaw:", yawInput);
    eulerLayout->addRow("Convention:", eulerConventionCombo);
    eulerLayout->addRow("", setEulerButton);
    
    // Add groups to attitude tab
    attitudeLayout->addWidget(quatGroup);
    attitudeLayout->addWidget(eulerGroup);
    attitudeLayout->addStretch();
    
    // ---- Vector Visualization Tab ----
    QWidget *vectorTab = new QWidget();
    QVBoxLayout *vectorLayout = new QVBoxLayout(vectorTab);
    
    // Add vector controls (will be created in a separate function)
    createVectorControls();
    
    // Add the vector controls group box to the vector tab
    QGroupBox *vectorControlsGroup = new QGroupBox("Vector Controls", vectorTab);
    QVBoxLayout *vectorControlsLayout = new QVBoxLayout(vectorControlsGroup);
    
    // Add each vector control
    for (int vectorType = AttitudeEngine::BODY_X; vectorType <= AttitudeEngine::ANGULAR_MOMENTUM; ++vectorType) {
        if (vectorControls.contains(vectorType)) {
            QHBoxLayout *row = new QHBoxLayout();
            
            // Get the vector name
            QString vectorName;
            switch (vectorType) {
                case AttitudeEngine::BODY_X: vectorName = "Body X Axis"; break;
                case AttitudeEngine::BODY_Y: vectorName = "Body Y Axis"; break;
                case AttitudeEngine::BODY_Z: vectorName = "Body Z Axis"; break;
                case AttitudeEngine::NADIR: vectorName = "Nadir Vector"; break;
                case AttitudeEngine::SUN: vectorName = "Sun Vector"; break;
                case AttitudeEngine::VELOCITY: vectorName = "Velocity Vector"; break;
                case AttitudeEngine::ANGULAR_MOMENTUM: vectorName = "Angular Momentum"; break;
                default: vectorName = "Unknown Vector"; break;
            }
            
            // Add label
            QLabel *label = new QLabel(vectorName);
            label->setMinimumWidth(120);
            row->addWidget(label);
            
            // Add enable checkbox
            row->addWidget(vectorControls[vectorType].enableCheckbox);
            
            // Add color button
            row->addWidget(vectorControls[vectorType].colorButton);
            
            // Add scale spinbox with a label
            QLabel *scaleLabel = new QLabel("Scale:");
            row->addWidget(scaleLabel);
            row->addWidget(vectorControls[vectorType].scaleSpinBox);
            
            // Add row to layout
            vectorControlsLayout->addLayout(row);
        }
    }
    
    vectorControlsLayout->addStretch();
    vectorLayout->addWidget(vectorControlsGroup);
    
    // Add tabs to the tabbed widget
    controlTabs->addTab(simTab, "Simulation");
    controlTabs->addTab(attitudeTab, "Attitude");
    controlTabs->addTab(vectorTab, "Vectors");
    
    // Add the tabbed widget to the control panel
    controlLayout->addWidget(controlTabs);
    
    // Create a splitter for resizable panels
    QSplitter *splitter = new QSplitter(Qt::Horizontal, ui->centralwidget);
    splitter->addWidget(controlPanel);
    splitter->addWidget(spacecraftView);
    splitter->setStretchFactor(0, 1);  // Control panel (20%)
    splitter->setStretchFactor(1, 4);  // Spacecraft view (80%)
    
    // Set splitter as the main widget
    QHBoxLayout *mainLayout = new QHBoxLayout(ui->centralwidget);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->addWidget(splitter);
    
    // Connect the load model button
    connect(loadModelButton, &QPushButton::clicked, this, &MainWindow::onLoadSpacecraftModel);
    
    // Connect the load mock data button
    connect(loadMockDataButton, &QPushButton::clicked, this, &MainWindow::onLoadMockData);
}

void MainWindow::createVectorControls()
{
    // Create control widgets for each vector type
    for (int vectorType = AttitudeEngine::BODY_X; vectorType <= AttitudeEngine::ANGULAR_MOMENTUM; ++vectorType) {
        VectorControlWidgets widgets;
        
        // Create enable checkbox
        widgets.enableCheckbox = new QCheckBox("Show");
        
        // Set initial state based on vector type
        switch (vectorType) {
            case AttitudeEngine::BODY_X:
            case AttitudeEngine::BODY_Y:
            case AttitudeEngine::BODY_Z:
            case AttitudeEngine::SUN:
            case AttitudeEngine::NADIR:
                widgets.enableCheckbox->setChecked(true);
                break;
            default:
                widgets.enableCheckbox->setChecked(false);
                break;
        }
        
        // Create color button
        widgets.colorButton = new QPushButton("Color");
        
        // Set initial color based on vector type
        switch (vectorType) {
            case AttitudeEngine::BODY_X:
                widgets.color = QColor(255, 0, 0); // Red
                break;
            case AttitudeEngine::BODY_Y:
                widgets.color = QColor(0, 255, 0); // Green
                break;
            case AttitudeEngine::BODY_Z:
                widgets.color = QColor(0, 0, 255); // Blue
                break;
            case AttitudeEngine::NADIR:
                widgets.color = QColor(255, 255, 0); // Yellow
                break;
            case AttitudeEngine::SUN:
                widgets.color = QColor(255, 165, 0); // Orange
                break;
            case AttitudeEngine::VELOCITY:
                widgets.color = QColor(0, 255, 255); // Cyan
                break;
            case AttitudeEngine::ANGULAR_MOMENTUM:
                widgets.color = QColor(255, 0, 255); // Magenta
                break;
            default:
                widgets.color = QColor(200, 200, 200); // Light gray
                break;
        }
        
        // Set color as button background
        QPalette pal = widgets.colorButton->palette();
        pal.setColor(QPalette::Button, widgets.color);
        widgets.colorButton->setAutoFillBackground(true);
        widgets.colorButton->setPalette(pal);
        widgets.colorButton->setFlat(true);
        
        // Create scale spinbox
        widgets.scaleSpinBox = new QDoubleSpinBox();
        widgets.scaleSpinBox->setRange(0.1, 10.0);
        widgets.scaleSpinBox->setSingleStep(0.1);
        widgets.scaleSpinBox->setValue(2.0);
        
        // Connect signals for this vector type
        connect(widgets.enableCheckbox, &QCheckBox::toggled, 
                [this, vectorType](bool checked) { onVectorToggled(vectorType, checked); });
        
        connect(widgets.colorButton, &QPushButton::clicked,
                [this, vectorType]() { onVectorColorChanged(vectorType); });
        
        connect(widgets.scaleSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                [this, vectorType](double value) { onVectorScaleChanged(vectorType, value); });
        
        // Store the widgets
        vectorControls[vectorType] = widgets;
    }
}

void MainWindow::setupConnections()
{
    // Connect attitude engine signals to main window
    connect(attitudeEngine, &AttitudeEngine::simulationTimeChanged, 
            this, &MainWindow::onSimulationTimeChanged);
    
    // Connect UI signals
    connect(startButton, &QPushButton::clicked, this, &MainWindow::onStartSimulation);
    connect(stopButton, &QPushButton::clicked, this, &MainWindow::onStopSimulation);
    connect(speedSlider, &QSlider::valueChanged, this, &MainWindow::onSimulationSpeedChanged);
    connect(dateTimeEdit, &QDateTimeEdit::dateTimeChanged, this, &MainWindow::onDateTimeChanged);
    
    // Connect quaternion input button
    connect(setQuatButton, &QPushButton::clicked, this, &MainWindow::onManualQuaternionSet);
    
    // Connect euler angles input button
    connect(setEulerButton, &QPushButton::clicked, this, &MainWindow::onEulerAnglesSet);
    
    // Connect reference frame combo box
    connect(referenceFrameCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onReferenceFrameChanged);
}

void MainWindow::createActions()
{
    // File menu
    QMenu *fileMenu = menuBar()->addMenu(tr("&File"));
    
    QAction *loadModelAction = new QAction(tr("&Load Spacecraft Model..."), this);
    connect(loadModelAction, &QAction::triggered, this, &MainWindow::onLoadSpacecraftModel);
    fileMenu->addAction(loadModelAction);
    
    QAction *loadDataAction = new QAction(tr("Load &Mock Data..."), this);
    connect(loadDataAction, &QAction::triggered, this, &MainWindow::onLoadMockData);
    fileMenu->addAction(loadDataAction);
    
    fileMenu->addSeparator();
    
    QAction *exitAction = new QAction(tr("E&xit"), this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QMainWindow::close);
    fileMenu->addAction(exitAction);
    
    // View menu
    QMenu *viewMenu = menuBar()->addMenu(tr("&View"));
    
    QAction *resetViewAction = new QAction(tr("&Reset View"), this);
    resetViewAction->setShortcut(QKeySequence(Qt::Key_R));
    connect(resetViewAction, &QAction::triggered, [this]() {
        // TODO: Implement view reset
        statusBar()->showMessage(tr("View reset"), 2000);
    });
    viewMenu->addAction(resetViewAction);
    
    // Simulation menu
    QMenu *simMenu = menuBar()->addMenu(tr("&Simulation"));
    
    QAction *startSimAction = new QAction(tr("&Start"), this);
    startSimAction->setShortcut(QKeySequence(Qt::Key_Space));
    connect(startSimAction, &QAction::triggered, this, &MainWindow::onStartSimulation);
    simMenu->addAction(startSimAction);
    
    QAction *stopSimAction = new QAction(tr("S&top"), this);
    stopSimAction->setShortcut(QKeySequence(Qt::Key_Escape));
    connect(stopSimAction, &QAction::triggered, this, &MainWindow::onStopSimulation);
    simMenu->addAction(stopSimAction);
}

void MainWindow::onStartSimulation()
{
    if (attitudeEngine && spacecraftView) {
        attitudeEngine->startSimulation();
        spacecraftView->startAnimation();
        statusBar()->showMessage(tr("Simulation started"), 2000);
    }
}

void MainWindow::onStopSimulation()
{
    if (attitudeEngine && spacecraftView) {
        attitudeEngine->stopSimulation();
        spacecraftView->stopAnimation();
        statusBar()->showMessage(tr("Simulation stopped"), 2000);
    }
}

void MainWindow::onSimulationSpeedChanged(int value)
{
    if (attitudeEngine && spacecraftView) {
        // Convert slider value (1-100) to a meaningful simulation speed
        // Use a more dramatic range from 0.1 to 20.0 instead of 0.1 to 10.0
        double speed = value / 5.0;  // 0.2 to 20.0
        
        // For very small values, ensure we don't go too slow
        if (speed < 0.2) {
            speed = 0.2;
        }
        
        attitudeEngine->setSimulationSpeed(speed);
        spacecraftView->setAnimationSpeed(speed);
        
        // Update status bar with the new speed
        statusBar()->showMessage(tr("Simulation speed: %1x").arg(speed), 2000);
    }
}

void MainWindow::onSimulationTimeChanged(const QString &timeString)
{
    // Update the time display in the UI
    if (timeDisplay) {
        timeDisplay->setText(timeString);
    }
}

void MainWindow::onDateTimeChanged(const QDateTime &dateTime)
{
    if (attitudeEngine && spacecraftView) {
        attitudeEngine->setSimulationDate(dateTime);
        spacecraftView->setDateTime(dateTime);
    }
}

void MainWindow::onLoadSpacecraftModel()
{
    QString filePath = QFileDialog::getOpenFileName(this,
                                                   tr("Load Spacecraft Model"),
                                                   QString(),
                                                   tr("OBJ Models (*.obj)"));
    
    if (!filePath.isEmpty()) {
        if (spacecraftView) {
            if (spacecraftView->loadSpacecraftModel(filePath)) {
                statusBar()->showMessage(tr("Loaded spacecraft model: %1").arg(filePath), 3000);
            } else {
                QMessageBox::warning(this, tr("Error"), tr("Failed to load spacecraft model."));
            }
        }
    }
}

void MainWindow::onVectorToggled(int vectorType, bool checked)
{
    if (spacecraftView) {
        spacecraftView->showVector(static_cast<AttitudeEngine::VectorType>(vectorType), checked);
        
        // Get vector name for status message
        QString vectorName;
        switch (vectorType) {
            case AttitudeEngine::BODY_X: vectorName = "Body X Axis"; break;
            case AttitudeEngine::BODY_Y: vectorName = "Body Y Axis"; break;
            case AttitudeEngine::BODY_Z: vectorName = "Body Z Axis"; break;
            case AttitudeEngine::NADIR: vectorName = "Nadir Vector"; break;
            case AttitudeEngine::SUN: vectorName = "Sun Vector"; break;
            case AttitudeEngine::VELOCITY: vectorName = "Velocity Vector"; break;
            case AttitudeEngine::ANGULAR_MOMENTUM: vectorName = "Angular Momentum"; break;
            default: vectorName = "Unknown Vector"; break;
        }
        
        statusBar()->showMessage(tr("%1: %2").arg(vectorName).arg(checked ? "Visible" : "Hidden"), 2000);
    }
}

void MainWindow::onVectorColorChanged(int vectorType)
{
    if (!vectorControls.contains(vectorType)) {
        return;
    }
    
    QColor currentColor = vectorControls[vectorType].color;
    QColor newColor = QColorDialog::getColor(currentColor, this, tr("Select Vector Color"));
    
    if (newColor.isValid()) {
        // Update stored color
        vectorControls[vectorType].color = newColor;
        
        // Update button color
        QPalette pal = vectorControls[vectorType].colorButton->palette();
        pal.setColor(QPalette::Button, newColor);
        vectorControls[vectorType].colorButton->setPalette(pal);
        
        // Update vector in view
        if (spacecraftView) {
            spacecraftView->setVectorColor(static_cast<AttitudeEngine::VectorType>(vectorType), newColor);
        }
    }
}

void MainWindow::onVectorScaleChanged(int vectorType, double scale)
{
    if (spacecraftView) {
        spacecraftView->setVectorScale(static_cast<AttitudeEngine::VectorType>(vectorType), scale);
    }
}

void MainWindow::onManualQuaternionSet()
{
    if (!attitudeEngine) {
        return;
    }
    
    // Get values from input fields
    bool ok;
    double w = quatWInput->text().toDouble(&ok);
    if (!ok) {
        QMessageBox::warning(this, tr("Input Error"), tr("Invalid value for quaternion W component"));
        return;
    }
    
    double x = quatXInput->text().toDouble(&ok);
    if (!ok) {
        QMessageBox::warning(this, tr("Input Error"), tr("Invalid value for quaternion X component"));
        return;
    }
    
    double y = quatYInput->text().toDouble(&ok);
    if (!ok) {
        QMessageBox::warning(this, tr("Input Error"), tr("Invalid value for quaternion Y component"));
        return;
    }
    
    double z = quatZInput->text().toDouble(&ok);
    if (!ok) {
        QMessageBox::warning(this, tr("Input Error"), tr("Invalid value for quaternion Z component"));
        return;
    }
    
    // Create quaternion
    Eigen::Quaterniond quat(w, x, y, z);
    
    // Normalize (in case user input is not normalized)
    quat.normalize();
    
    // Update the input fields with normalized values
    quatWInput->setText(QString::number(quat.w(), 'f', 6));
    quatXInput->setText(QString::number(quat.x(), 'f', 6));
    quatYInput->setText(QString::number(quat.y(), 'f', 6));
    quatZInput->setText(QString::number(quat.z(), 'f', 6));
    
    // Set quaternion in attitude engine
    attitudeEngine->setQuaternion(quat);
    
    statusBar()->showMessage(tr("Quaternion set: [%1, %2, %3, %4]")
                           .arg(quat.w(), 0, 'f', 4)
                           .arg(quat.x(), 0, 'f', 4)
                           .arg(quat.y(), 0, 'f', 4)
                           .arg(quat.z(), 0, 'f', 4), 3000);
}

void MainWindow::onEulerAnglesSet()
{
    if (!attitudeEngine) {
        return;
    }
    
    // Get values from input fields
    double roll = rollInput->value() * M_PI / 180.0;   // Convert to radians
    double pitch = pitchInput->value() * M_PI / 180.0; // Convert to radians
    double yaw = yawInput->value() * M_PI / 180.0;     // Convert to radians
    
    // Get convention
    QString convention = eulerConventionCombo->currentText().split(" ").first();
    
    // Convert to quaternion
    Eigen::Quaterniond quat = eulerToQuaternion(roll, pitch, yaw, convention);
    
    // Set quaternion in attitude engine
    attitudeEngine->setQuaternion(quat);
    
    // Update quaternion input fields
    quatWInput->setText(QString::number(quat.w(), 'f', 6));
    quatXInput->setText(QString::number(quat.x(), 'f', 6));
    quatYInput->setText(QString::number(quat.y(), 'f', 6));
    quatZInput->setText(QString::number(quat.z(), 'f', 6));
    
    statusBar()->showMessage(tr("Euler angles set: Roll=%1°, Pitch=%2°, Yaw=%3° (%4)")
                           .arg(rollInput->value(), 0, 'f', 2)
                           .arg(pitchInput->value(), 0, 'f', 2)
                           .arg(yawInput->value(), 0, 'f', 2)
                           .arg(convention), 3000);
}

void MainWindow::onReferenceFrameChanged(int index)
{
    if (!attitudeEngine) {
        return;
    }
    
    QString frameName;
    switch (index) {
        case 0: frameName = "ECI"; break;
        case 1: frameName = "ECEF"; break;
        case 2: frameName = "LVLH"; break;
        default: frameName = "ECI"; break;
    }
    
    attitudeEngine->setReferenceFrame(frameName);
    statusBar()->showMessage(tr("Reference frame changed to %1").arg(frameName), 2000);
}

void MainWindow::onLoadMockData()
{
    QString filePath = QFileDialog::getOpenFileName(this,
                                                   tr("Load Mock Telemetry Data"),
                                                   "data/mock",  // Default directory
                                                   tr("JSON Files (*.json)"));
    
    if (!filePath.isEmpty() && attitudeEngine) {
        attitudeEngine->loadMockData(filePath);
        statusBar()->showMessage(tr("Loaded mock data: %1").arg(filePath), 3000);
    }
}

Eigen::Quaterniond MainWindow::eulerToQuaternion(double roll, double pitch, double yaw, const QString &convention)
{
    Eigen::Quaterniond q;
    
    if (convention == "XYZ") {
        // Roll (X), Pitch (Y), Yaw (Z) convention
        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
        
        q = yawAngle * pitchAngle * rollAngle;
    }
    else if (convention == "ZYX") {
        // Yaw (Z), Pitch (Y), Roll (X) convention (aircraft)
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
        
        q = rollAngle * pitchAngle * yawAngle;
    }
    else if (convention == "ZXZ") {
        // Aerospace convention
        Eigen::AngleAxisd z1Angle(yaw, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd xAngle(pitch, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd z2Angle(roll, Eigen::Vector3d::UnitZ());
        
        q = z1Angle * xAngle * z2Angle;
    }
    else {
        // Default to XYZ if not recognized
        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
        
        q = yawAngle * pitchAngle * rollAngle;
    }
    
    return q.normalized();
}