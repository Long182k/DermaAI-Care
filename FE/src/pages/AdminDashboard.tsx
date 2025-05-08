import { useQuery } from "@tanstack/react-query";
import { statisticsApi } from "@/api/statistics";
import { Navbar } from "@/components/Navbar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Loader2,
  Users,
  UserCheck,
  FileImage,
  Calendar,
  CreditCard,
  TrendingUp,
  Clock,
  Calendar as CalendarIcon,
  Clock as ClockIcon,
} from "lucide-react";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAppStore } from "@/store";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line, Bar, Pie, Doughnut } from "react-chartjs-2";

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

// Chart options
const lineChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: "top" as const,
    },
  },
};

const barChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: "top" as const,
    },
  },
};

const pieChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: "right" as const,
    },
  },
};

// Add doughnutOptions (same as pieChartOptions)
const doughnutOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: "right" as const,
    },
  },
};

// Helper function to generate random colors
const generateColors = (count: number) => {
  const colors = [];
  for (let i = 0; i < count; i++) {
    const r = Math.floor(Math.random() * 200);
    const g = Math.floor(Math.random() * 200);
    const b = Math.floor(Math.random() * 200);
    colors.push(`rgba(${r}, ${g}, ${b}, 0.7)`);
  }
  return colors;
};

const AdminDashboard = () => {
  const { userInfo } = useAppStore();
  const navigate = useNavigate();

  // Redirect if not admin
  useEffect(() => {
    if (userInfo && userInfo.role !== "ADMIN") {
      navigate("/");
    }
  }, [userInfo, navigate]);

  // Fetch overview statistics
  const {
    data: overviewData,
    isLoading: isLoadingOverview,
    error: overviewError,
  } = useQuery({
    queryKey: ["statistics", "overview"],
    queryFn: statisticsApi.getOverview,
  });

  // Fetch patient statistics
  const {
    data: patientData,
    isLoading: isLoadingPatients,
    error: patientsError,
  } = useQuery({
    queryKey: ["statistics", "patients"],
    queryFn: statisticsApi.getPatients,
  });

  // Fetch doctor statistics
  const {
    data: doctorData,
    isLoading: isLoadingDoctors,
    error: doctorsError,
  } = useQuery({
    queryKey: ["statistics", "doctors"],
    queryFn: statisticsApi.getDoctors,
  });

  // Fetch prediction statistics
  const {
    data: predictionData,
    isLoading: isLoadingPredictions,
    error: predictionsError,
  } = useQuery({
    queryKey: ["statistics", "predictions"],
    queryFn: statisticsApi.getPredictions,
  });

  // Fetch appointment statistics
  const {
    data: appointmentData,
    isLoading: isLoadingAppointments,
    error: appointmentsError,
  } = useQuery({
    queryKey: ["statistics", "appointments"],
    queryFn: statisticsApi.getAppointments,
  });

  const isLoading =
    isLoadingOverview ||
    isLoadingPatients ||
    isLoadingDoctors ||
    isLoadingPredictions ||
    isLoadingAppointments;

  const hasError =
    overviewError ||
    patientsError ||
    doctorsError ||
    predictionsError ||
    appointmentsError;

  // Prepare chart data for overview
  const prepareMonthlyOverviewData = () => {
    if (!overviewData?.charts?.monthly) return null;

    const { labels, datasets } = overviewData.charts.monthly;

    return {
      labels,
      datasets: datasets.map((dataset, index) => ({
        label: dataset.label,
        data: dataset.data,
        borderColor:
          index === 0
            ? "rgba(75, 192, 192, 1)"
            : index === 1
            ? "rgba(153, 102, 255, 1)"
            : index === 2
            ? "rgba(255, 159, 64, 1)"
            : "rgba(255, 99, 132, 1)",
        backgroundColor:
          index === 0
            ? "rgba(75, 192, 192, 0.2)"
            : index === 1
            ? "rgba(153, 102, 255, 0.2)"
            : index === 2
            ? "rgba(255, 159, 64, 0.2)"
            : "rgba(255, 99, 132, 0.2)",
        borderWidth: 2,
        tension: 0.3,
      })),
    };
  };

  // Prepare patient gender distribution chart data
  const prepareGenderDistributionData = (data) => {
    if (!data?.summary?.genderDistribution) return null;

    const genderData = data.summary.genderDistribution;
    const labels = [];
    const values = [];

    genderData.forEach((item) => {
      labels.push(item.gender);
      values.push(item.count);
    });

    return {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: generateColors(values.length),
          borderWidth: 1,
        },
      ],
    };
  };

  // Prepare monthly registrations chart data
  const prepareMonthlyRegistrationsData = (data) => {
    if (!data?.charts?.monthlyRegistrations) return null;

    const { labels, data: monthlyData } = data.charts.monthlyRegistrations;

    return {
      labels,
      datasets: [
        {
          label: "New Registrations",
          data: monthlyData,
          backgroundColor: "rgba(75, 192, 192, 0.5)",
          borderColor: "rgba(75, 192, 192, 1)",
          borderWidth: 1,
        },
      ],
    };
  };

  // Prepare doctors by experience chart data
  const prepareDoctorsByExperienceData = (data) => {
    if (!data?.charts?.doctorsByExperience) return null;

    const experienceData = data.charts.doctorsByExperience;
    const labels = experienceData.map((item) => `${item.experience} years`);
    const values = experienceData.map((item) => item.count);

    return {
      labels,
      datasets: [
        {
          label: "Doctors",
          data: values,
          backgroundColor: "rgba(153, 102, 255, 0.5)",
          borderColor: "rgba(153, 102, 255, 1)",
          borderWidth: 1,
        },
      ],
    };
  };

  // Prepare predictions by status chart data
  const preparePredictionsByStatusData = (data) => {
    if (!data?.summary?.predictionsByStatus) return null;

    const statusData = data.summary.predictionsByStatus;
    const labels = statusData.map((item) => item.status);
    const values = statusData.map((item) => item.count);

    return {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: generateColors(values.length),
          borderWidth: 1,
        },
      ],
    };
  };

  // Prepare monthly predictions chart data
  const prepareMonthlyPredictionsData = (data) => {
    if (!data?.charts?.monthlyPredictions) return null;

    const { labels, data: monthlyData } = data.charts.monthlyPredictions;

    return {
      labels,
      datasets: [
        {
          label: "Predictions",
          data: monthlyData,
          backgroundColor: "rgba(255, 159, 64, 0.5)",
          borderColor: "rgba(255, 159, 64, 1)",
          borderWidth: 1,
        },
      ],
    };
  };

  // Prepare appointments by status chart data
  const prepareAppointmentsByStatusData = (data) => {
    if (!data?.summary?.appointmentsByStatus) return null;

    const statusData = data.summary.appointmentsByStatus;
    const labels = statusData.map((item) => item.status);
    const values = statusData.map((item) => item.count);

    return {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: generateColors(values.length),
          borderWidth: 1,
        },
      ],
    };
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container py-20">
          <div className="flex justify-center items-center py-20">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <span className="ml-2">Loading dashboard data...</span>
          </div>
        </main>
      </div>
    );
  }

  if (hasError) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container py-20">
          <div className="text-center text-red-500 py-10">
            Failed to load dashboard data. Please try again later.
          </div>
        </main>
      </div>
    );
  }

  // Prepare data for charts
  const predictionStatusData = preparePredictionsByStatusData(predictionData);
  const monthlyPredictionData = prepareMonthlyPredictionsData(predictionData);
  const appointmentStatusData =
    prepareAppointmentsByStatusData(appointmentData);
  const monthlyAppointmentData = {
    labels: appointmentData?.charts?.monthlyAppointments?.labels || [],
    datasets: [
      {
        label: "Appointments",
        data: appointmentData?.charts?.monthlyAppointments?.data || [],
        backgroundColor: "rgba(34, 197, 94, 0.5)",
        borderColor: "rgba(34, 197, 94, 1)",
        borderWidth: 1,
      },
    ],
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container pt-24 pb-10">
        <h1 className="text-3xl font-bold mb-6">Admin Dashboard</h1>

        {/* Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
          <Card className="shadow-md hover:shadow-lg transition-shadow">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Total Patients
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center">
                <Users className="h-5 w-5 text-blue-500 mr-2" />
                <span className="text-2xl font-bold">
                  {overviewData?.summary.totalPatients || 0}
                </span>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-md hover:shadow-lg transition-shadow">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Total Doctors
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center">
                <UserCheck className="h-5 w-5 text-purple-500 mr-2" />
                <span className="text-2xl font-bold">
                  {overviewData?.summary.totalDoctors || 0}
                </span>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-md hover:shadow-lg transition-shadow">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Total Predictions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center">
                <FileImage className="h-5 w-5 text-orange-500 mr-2" />
                <span className="text-2xl font-bold">
                  {overviewData?.summary.totalPredictions || 0}
                </span>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-md hover:shadow-lg transition-shadow">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Total Appointments
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center">
                <Calendar className="h-5 w-5 text-green-500 mr-2" />
                <span className="text-2xl font-bold">
                  {overviewData?.summary.totalAppointments || 0}
                </span>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-md hover:shadow-lg transition-shadow">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Total Payments
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center">
                <CreditCard className="h-5 w-5 text-red-500 mr-2" />
                <span className="text-2xl font-bold">
                  {overviewData?.summary.totalPayments || 0}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Overview Chart */}
        <Card className="mb-8 shadow-md">
          <CardHeader>
            <CardTitle>Monthly Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[400px]">
              {prepareMonthlyOverviewData() ? (
                <Line
                  options={lineChartOptions}
                  data={prepareMonthlyOverviewData()}
                />
              ) : (
                <div className="flex items-center justify-center h-full">
                  <p className="text-muted-foreground">No data available</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Detailed Statistics Tabs */}
        <Tabs defaultValue="patients" className="w-full">
          <TabsList className="grid grid-cols-4 mb-8">
            <TabsTrigger value="patients" className="text-sm md:text-base">
              Patients
            </TabsTrigger>
            <TabsTrigger value="doctors" className="text-sm md:text-base">
              Doctors
            </TabsTrigger>
            <TabsTrigger value="predictions" className="text-sm md:text-base">
              Predictions
            </TabsTrigger>
            <TabsTrigger value="appointments" className="text-sm md:text-base">
              Appointments
            </TabsTrigger>
          </TabsList>

          {/* Patients Tab */}
          <TabsContent value="patients">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="shadow-md">
                <CardHeader>
                  <CardTitle>Patient Gender Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    {prepareGenderDistributionData(patientData) ? (
                      <Doughnut
                        options={pieChartOptions}
                        data={prepareGenderDistributionData(patientData)}
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        <p className="text-muted-foreground">
                          No data available
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card className="shadow-md">
                <CardHeader>
                  <CardTitle>Monthly Patient Registrations</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    {prepareMonthlyRegistrationsData(patientData) ? (
                      <Bar
                        options={barChartOptions}
                        data={prepareMonthlyRegistrationsData(patientData)}
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        <p className="text-muted-foreground">
                          No data available
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card className="md:col-span-2 shadow-md">
                <CardHeader>
                  <CardTitle>Top Patients by Appointments</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px] overflow-auto">
                    {patientData?.charts.topPatientsByAppointments?.length ? (
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b bg-muted/50">
                              <th className="text-left py-3 px-4 font-medium">
                                Patient Name
                              </th>
                              <th className="text-right py-3 px-4 font-medium">
                                Appointments
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {patientData.charts.topPatientsByAppointments.map(
                              (patient, index) => (
                                <tr
                                  key={index}
                                  className="border-b hover:bg-muted/30 transition-colors"
                                >
                                  <td className="py-3 px-4">{patient.name}</td>
                                  <td className="text-right py-3 px-4">
                                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                      {patient.count}
                                    </span>
                                  </td>
                                </tr>
                              )
                            )}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        <p className="text-muted-foreground">
                          No data available
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Doctors Tab */}
          <TabsContent value="doctors">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="shadow-md">
                <CardHeader>
                  <CardTitle>Doctor Gender Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    {prepareGenderDistributionData(doctorData) ? (
                      <Pie
                        options={pieChartOptions}
                        data={prepareGenderDistributionData(doctorData)}
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        <p className="text-muted-foreground">
                          No data available
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card className="shadow-md">
                <CardHeader>
                  <CardTitle>Doctors by Experience</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    {prepareDoctorsByExperienceData(doctorData) ? (
                      <Bar
                        options={barChartOptions}
                        data={prepareDoctorsByExperienceData(doctorData)}
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        <p className="text-muted-foreground">
                          No data available
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card className="md:col-span-2 shadow-md">
                <CardHeader>
                  <CardTitle>Top Doctors by Appointments</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px] overflow-auto">
                    {doctorData?.charts.topDoctorsByAppointments?.length ? (
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b bg-muted/50">
                              <th className="text-left py-3 px-4 font-medium">
                                Doctor Name
                              </th>
                              <th className="text-right py-3 px-4 font-medium">
                                Appointments
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {doctorData.charts.topDoctorsByAppointments.map(
                              (doctor, index) => (
                                <tr
                                  key={index}
                                  className="border-b hover:bg-muted/30 transition-colors"
                                >
                                  <td className="py-3 px-4">{doctor.name}</td>
                                  <td className="text-right py-3 px-4">
                                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                                      {doctor.count}
                                    </span>
                                  </td>
                                </tr>
                              )
                            )}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        <p className="text-muted-foreground">
                          No data available
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Predictions Tab */}
          <TabsContent value="predictions">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="shadow-md">
                <CardHeader>
                  <CardTitle>Predictions by Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px] flex items-center justify-center">
                    {predictionData?.summary.predictionsByStatus ? (
                      <Doughnut
                        options={doughnutOptions}
                        data={predictionStatusData}
                      />
                    ) : (
                      <p className="text-muted-foreground">No data available</p>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card className="shadow-md">
                <CardHeader>
                  <CardTitle>Monthly Predictions</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px] flex items-center justify-center">
                    {predictionData?.charts?.monthlyPredictions ? (
                      <Bar
                        options={barChartOptions}
                        data={monthlyPredictionData}
                      />
                    ) : (
                      <p className="text-muted-foreground">No data available</p>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card className="md:col-span-2 shadow-md">
                <CardHeader>
                  <CardTitle>Recent Predictions</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-auto max-h-[400px]">
                    {predictionData?.summary.predictionsByStatus?.length ? (
                      <table className="w-full">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-3 px-4 font-semibold">
                              Status
                            </th>
                            <th className="text-right py-3 px-4 font-semibold">
                              Count
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {predictionData.summary.predictionsByStatus.map(
                            (item, index) => (
                              <tr
                                key={index}
                                className="border-b hover:bg-muted/50 transition-colors"
                              >
                                <td className="py-3 px-4">
                                  <span
                                    className={`inline-flex items-center justify-center rounded-full px-2.5 py-0.5 text-xs font-medium
                                  ${
                                    item.status === "COMPLETED"
                                      ? "bg-green-100 text-green-800"
                                      : item.status === "FAILED"
                                      ? "bg-red-100 text-red-800"
                                      : item.status === "PROCESSING"
                                      ? "bg-blue-100 text-blue-800"
                                      : "bg-gray-100 text-gray-800"
                                  }`}
                                  >
                                    {item.status}
                                  </span>
                                </td>
                                <td className="text-right py-3 px-4">
                                  <span className="inline-flex items-center justify-center bg-primary/10 text-primary rounded-full px-2.5 py-0.5 text-sm font-medium">
                                    {item.count}
                                  </span>
                                </td>
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    ) : (
                      <p className="text-muted-foreground text-center py-8">
                        No data available
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Appointments Tab */}
          <TabsContent value="appointments">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="shadow-md">
                <CardHeader>
                  <CardTitle>Appointments by Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px] flex items-center justify-center">
                    {appointmentData?.summary.appointmentsByStatus ? (
                      <Doughnut
                        options={doughnutOptions}
                        data={appointmentStatusData}
                      />
                    ) : (
                      <p className="text-muted-foreground">No data available</p>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card className="shadow-md">
                <CardHeader>
                  <CardTitle>Monthly Appointments</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px] flex items-center justify-center">
                    {appointmentData?.charts?.monthlyAppointments ? (
                      <Bar
                        options={barChartOptions}
                        data={monthlyAppointmentData}
                      />
                    ) : (
                      <p className="text-muted-foreground">No data available</p>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card className="md:col-span-2 shadow-md">
                <CardHeader>
                  <CardTitle>Appointments by Day</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-auto max-h-[400px]">
                    {appointmentData?.charts?.appointmentsByDay?.length ? (
                      <table className="w-full">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-3 px-4 font-semibold">
                              Day
                            </th>
                            <th className="text-right py-3 px-4 font-semibold">
                              Count
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {appointmentData.charts.appointmentsByDay.map(
                            (item, index) => (
                              <tr
                                key={index}
                                className="border-b hover:bg-muted/50 transition-colors"
                              >
                                <td className="py-3 px-4">{item.day}</td>
                                <td className="text-right py-3 px-4">
                                  <span className="inline-flex items-center justify-center rounded-full px-2.5 py-0.5 text-xs font-medium bg-green-100 text-green-800">
                                    {item.count}
                                  </span>
                                </td>
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    ) : (
                      <p className="text-muted-foreground text-center py-8">
                        No data available
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default AdminDashboard;
