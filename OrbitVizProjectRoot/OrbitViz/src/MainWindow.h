#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QMap>
#include <QVector>
#include <Eigen/Geometry>

// Forward declarations
class AttitudeEngine;
class SpacecraftView;
class QLabel;
class QSlider;
class QDateTimeEdit;
class QPushButton;
class QComboBox;
class QCheckBox;
class QLineEdit;
class QDoubleSpinBox;
class QTabWidget;
class QGroupBox;
class QColorDialog;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onStartSimulation();
    void onStopSimulation();
    void onSimulationSpeedChanged(int value);
    void onSimulationTimeChanged(const QString &timeString);
    void onDateTimeChanged(const QDateTime &dateTime);
    void onLoadSpacecraftModel();
    
    // Vector control slots
    void onVectorToggled(int vectorType, bool checked);
    void onVectorColorChanged(int vectorType);
    void onVectorScaleChanged(int vectorType, double scale);
    
    // Quaternion input slots
    void onManualQuaternionSet();
    void onEulerAnglesSet();
    
    // Reference frame selection
    void onReferenceFrameChanged(int index);
    
    // Mock data loading
    void onLoadMockData();

private:
    void setupUI();
    void createActions();
    void setupConnections();
    void createVectorControls();

    Ui::MainWindow *ui;
    SpacecraftView *spacecraftView;
    AttitudeEngine *attitudeEngine;
    
    // UI elements
    QLabel *timeDisplay;
    QSlider *speedSlider;
    QDateTimeEdit *dateTimeEdit;
    QPushButton *startButton;
    QPushButton *stopButton;
    QTabWidget *controlTabs;
    
    // Quaternion input widgets
    QLineEdit *quatWInput;
    QLineEdit *quatXInput;
    QLineEdit *quatYInput;
    QLineEdit *quatZInput;
    QPushButton *setQuatButton;
    
    // Euler angle input widgets
    QDoubleSpinBox *rollInput;
    QDoubleSpinBox *pitchInput;
    QDoubleSpinBox *yawInput;
    QComboBox *eulerConventionCombo;
    QPushButton *setEulerButton;
    
    // Reference frame selection
    QComboBox *referenceFrameCombo;
    
    // Vector control widgets
    struct VectorControlWidgets {
        QCheckBox *enableCheckbox;
        QPushButton *colorButton;
        QDoubleSpinBox *scaleSpinBox;
        QColor color;
    };
    
    QMap<int, VectorControlWidgets> vectorControls;
    
    // Convert Euler angles to quaternion
    Eigen::Quaterniond eulerToQuaternion(double roll, double pitch, double yaw, const QString &convention);
};

#endif // MAINWINDOW_H