import type { EditUserNamesDto, PaymentStats } from "@/api/statistics";
import { statisticsApi } from "@/api/statistics";
import { Navbar } from "@/components/Navbar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAppStore } from "@/store";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
} from "chart.js";
import {
  Calendar,
  CreditCard,
  FileImage,
  Loader2,
  UserCheck,
  Users,
} from "lucide-react";
import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Bar, Doughnut, Line, Pie } from "react-chartjs-2";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";

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

  // Add state for the edit name modal
  const [editNameModalOpen, setEditNameModalOpen] = useState(false);
  const [currentUserId, setCurrentUserId] = useState("");
  const [editNameForm, setEditNameForm] = useState({
    firstName: "",
    lastName: "",
  });

  // Function to open the edit modal
  const handleOpenEditModal = (
    userId: string,
    firstName: string,
    lastName: string
  ) => {
    setCurrentUserId(userId);
    setEditNameForm({
      firstName: firstName || "",
      lastName: lastName || "",
    });
    setEditNameModalOpen(true);
  };

  // Function to handle form submission
  const handleEditNameSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (currentUserId && (editNameForm.firstName || editNameForm.lastName)) {
      editUserNamesMutation.mutate({
        userId: currentUserId,
        dto: {
          firstName: editNameForm.firstName,
          lastName: editNameForm.lastName,
        },
      });
      setEditNameModalOpen(false);
    }
  };

  // Mutation for editing user names
  const editUserNamesMutation = useMutation({
    mutationFn: ({ userId, dto }: { userId: string; dto: EditUserNamesDto }) =>
      statisticsApi.editUserNames(userId, dto),
    onSuccess: () => {
      refetchPatients();
      refetchDoctors();
      toast.success("User names updated successfully");
      // Optionally refetch users/statistics here
    },
    onError: (error: any) => {
      toast.error(
        error?.response?.data?.message || "Failed to update user names"
      );
    },
  });

  // Mutation for changing user active status
  const changeUserActiveMutation = useMutation({
    mutationFn: ({ userId, isActive }: { userId: string; isActive: boolean }) =>
      statisticsApi.changeUserActive(userId, isActive),
    onSuccess: () => {
      toast.success("User active status updated");
      // Refetch all relevant data
      refetchPatients();
      refetchDoctors();
    },
    onError: (error: any) => {
      toast.error(
        error?.response?.data?.message || "Failed to update user status"
      );
    },
  });

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
    refetch: refetchPatients,
  } = useQuery({
    queryKey: ["statistics", "patients"],
    queryFn: statisticsApi.getPatients,
  });

  // Fetch doctor statistics
  const {
    data: doctorData,
    isLoading: isLoadingDoctors,
    error: doctorsError,
    refetch: refetchDoctors,
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

  // Fetch payment statistics
  const {
    data: paymentData,
    isLoading: isLoadingPayments,
    error: paymentsError,
  } = useQuery({
    queryKey: ["statistics", "payments"],
    queryFn: statisticsApi.getPayments,
  });

  const isLoading =
    isLoadingOverview ||
    isLoadingPatients ||
    isLoadingDoctors ||
    isLoadingPredictions ||
    isLoadingAppointments ||
    isLoadingPayments;

  const hasError =
    overviewError ||
    patientsError ||
    doctorsError ||
    predictionsError ||
    appointmentsError ||
    paymentsError;

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

  // Prepare data for Appointments by Day
  const prepareAppointmentsByDayData = (data) => {
    if (!data?.charts?.appointmentsByDay) return null;
    const labels = data.charts.appointmentsByDay.map((item) => item.day);
    const counts = data.charts.appointmentsByDay.map((item) => item.count);
    return {
      labels,
      datasets: [
        {
          label: "Appointments",
          data: counts,
          backgroundColor: "rgba(34, 197, 94, 0.5)",
          borderColor: "rgba(34, 197, 94, 1)",
          borderWidth: 1,
        },
      ],
    };
  };

  const appointmentsByDayData = prepareAppointmentsByDayData(appointmentData);

  // Prepare payments by day chart data
  const preparePaymentsByDayData = (data: PaymentStats | undefined) => {
    if (!data?.charts?.paymentsByDay) return null;
    const labels = data.charts.paymentsByDay.map((item) => item.day);
    const counts = data.charts.paymentsByDay.map((item) => item.count);
    return {
      labels,
      datasets: [
        {
          label: "Payments",
          data: counts,
          backgroundColor: "rgba(251, 191, 36, 0.5)",
          borderColor: "rgba(251, 191, 36, 1)",
          borderWidth: 1,
        },
      ],
    };
  };
  const paymentsByDayData = preparePaymentsByDayData(paymentData);

  // Prepare monthly payments chart data
  const prepareMonthlyPaymentsData = (data: PaymentStats | undefined) => {
    if (!data?.charts?.monthlyPayments) return null;
    const { labels, data: monthlyData } = data.charts.monthlyPayments;
    return {
      labels,
      datasets: [
        {
          label: "Payments",
          data: monthlyData,
          backgroundColor: "rgba(251, 191, 36, 0.5)",
          borderColor: "rgba(251, 191, 36, 1)",
          borderWidth: 1,
        },
      ],
    };
  };
  const monthlyPaymentsData = prepareMonthlyPaymentsData(paymentData);

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
          <TabsList className="grid grid-cols-5 mb-8">
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
            <TabsTrigger value="payments" className="text-sm md:text-base">
              Payments
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
                              <th className="text-right py-3 px-4 font-medium">
                                Actions
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
                                  <td className="text-right py-3 px-4">
                                    <div className="flex gap-2 justify-end">
                                      <button
                                        className="bg-primary text-white px-3 py-1 rounded hover:bg-primary/80 focus:outline-none"
                                        aria-label="Edit Name"
                                        onClick={() =>
                                          handleOpenEditModal(
                                            patient.id,
                                            patient.firstName || "",
                                            patient.lastName || ""
                                          )
                                        }
                                        disabled={
                                          editUserNamesMutation.isPending
                                        }
                                      >
                                        {editUserNamesMutation.isPending
                                          ? "Saving..."
                                          : "Edit Name"}
                                      </button>
                                      <button
                                        className={`${
                                          patient.isActive
                                            ? "bg-red-500 hover:bg-red-600"
                                            : "bg-green-500 hover:bg-green-600"
                                        } text-white px-3 py-1 rounded focus:outline-none`}
                                        aria-label="Toggle Active"
                                        onClick={() => {
                                          changeUserActiveMutation.mutate({
                                            userId: patient.id,
                                            isActive: !patient.isActive,
                                          });
                                        }}
                                        disabled={
                                          changeUserActiveMutation.isPending
                                        }
                                      >
                                        {patient.isActive
                                          ? "Deactivate"
                                          : "Activate"}
                                      </button>
                                    </div>
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
                              <th className="text-right py-3 px-4 font-medium">
                                Actions
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
                                  <td className="py-3 px-4">{`Dr. ${doctor.name}`}</td>
                                  <td className="text-right py-3 px-4">
                                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                                      {doctor.count}
                                    </span>
                                  </td>
                                  <td className="text-right py-3 px-4">
                                    <div className="flex gap-2 justify-end">
                                      <button
                                        className="bg-primary text-white px-3 py-1 rounded hover:bg-primary/80 focus:outline-none"
                                        aria-label="Edit Name"
                                        onClick={() =>
                                          handleOpenEditModal(
                                            doctor.id,
                                            doctor.firstName || "",
                                            doctor.lastName || ""
                                          )
                                        }
                                        disabled={
                                          editUserNamesMutation.isPending
                                        }
                                      >
                                        {editUserNamesMutation.isPending
                                          ? "Saving..."
                                          : "Edit Name"}
                                      </button>
                                      <button
                                        className={`${
                                          doctor.isActive
                                            ? "bg-red-500 hover:bg-red-600"
                                            : "bg-green-500 hover:bg-green-600"
                                        } text-white px-3 py-1 rounded focus:outline-none`}
                                        aria-label="Toggle Active"
                                        onClick={() => {
                                          changeUserActiveMutation.mutate({
                                            userId: doctor.id,
                                            isActive: !doctor.isActive,
                                          });
                                        }}
                                        disabled={
                                          changeUserActiveMutation.isPending
                                        }
                                      >
                                        {doctor.isActive
                                          ? "Deactivate"
                                          : "Activate"}
                                      </button>
                                    </div>
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
                  <CardTitle>Skin Lesion Prediction Type</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px] flex items-center justify-center">
                    {predictionData?.charts?.lesionDistribution ? (
                      <Doughnut
                        options={doughnutOptions}
                        data={{
                          labels: predictionData.charts.lesionDistribution.map(
                            (item) => item.name
                          ),
                          datasets: [
                            {
                              data: predictionData.charts.lesionDistribution.map(
                                (item) => item.count
                              ),
                              backgroundColor: [
                                "#f87171",
                                "#60a5fa",
                                "#fbbf24",
                                "#34d399",
                                "#a78bfa",
                                "#f472b6",
                                "#38bdf8",
                                "#facc15",
                              ],
                            },
                          ],
                        }}
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
                  <CardTitle>Recent Lesion Type Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-auto max-h-[400px]">
                    {predictionData?.summary?.lesionDistribution?.length ? (
                      <table className="w-full">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-3 px-4 font-semibold">
                              Class
                            </th>
                            <th className="text-left py-3 px-4 font-semibold">
                              Name
                            </th>
                            <th className="text-left py-3 px-4 font-semibold">
                              Cancer Status
                            </th>
                            <th className="text-right py-3 px-4 font-semibold">
                              Count
                            </th>
                            <th className="text-right py-3 px-4 font-semibold">
                              Percent
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {predictionData.summary.lesionDistribution.map(
                            (item, index) => (
                              <tr
                                key={index}
                                className="border-b hover:bg-muted/50 transition-colors"
                              >
                                <td className="py-3 px-4 font-mono">
                                  {item.class}
                                </td>
                                <td className="py-3 px-4">{item.name}</td>
                                <td className="py-3 px-4">
                                  {item.isCancerous === true && (
                                    <span className="text-red-600 font-semibold ml-2">
                                      Cancerous
                                    </span>
                                  )}
                                  {item.isCancerous === false && (
                                    <span className="text-green-600 font-semibold ml-2">
                                      Non-cancerous
                                    </span>
                                  )}
                                  {item.isCancerous === null && (
                                    <span className="text-gray-500 font-semibold ml-2">
                                      Unknown
                                    </span>
                                  )}
                                </td>
                                <td className="text-right py-3 px-4">
                                  <span className="inline-flex items-center justify-center bg-primary/10 text-primary rounded-full px-2.5 py-0.5 text-sm font-medium">
                                    {item.count}
                                  </span>
                                </td>
                                <td className="text-right py-3 px-4">
                                  <span className="inline-flex items-center justify-center bg-primary/10 text-primary rounded-full px-2.5 py-0.5 text-sm font-medium">
                                    {item.percent}%
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
                  <div className="h-[300px] flex items-center justify-center">
                    {appointmentsByDayData ? (
                      <Bar
                        options={barChartOptions}
                        data={appointmentsByDayData}
                      />
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

          {/* Payments Tab */}
          <TabsContent value="payments">
            <div className="grid grid-cols-1 gap-6">
              <Card className="mb-8 shadow-md">
                <CardHeader>
                  <CardTitle>Payment Statistics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <div className="mb-4">
                        <div className="flex flex-wrap gap-4">
                          <div className="flex flex-col items-center">
                            <span className="text-lg font-semibold">
                              Total Payments
                            </span>
                            <span className="text-2xl font-bold">
                              {paymentData?.summary.totalPayments ?? 0}
                            </span>
                          </div>
                          <div className="flex flex-col items-center">
                            <span className="text-lg font-semibold">
                              Completed
                            </span>
                            <span className="text-2xl font-bold text-green-600">
                              {paymentData?.summary.completedPayments ?? 0}
                            </span>
                          </div>
                          <div className="flex flex-col items-center">
                            <span className="text-lg font-semibold">
                              Pending
                            </span>
                            <span className="text-2xl font-bold text-yellow-600">
                              {paymentData?.summary.pendingPayments ?? 0}
                            </span>
                          </div>
                          <div className="flex flex-col items-center">
                            <span className="text-lg font-semibold">
                              Total Completed Amount
                            </span>
                            <span className="text-2xl font-bold text-primary">
                              $
                              {paymentData?.summary.totalAmountCompleted?.toFixed(
                                2
                              ) ?? "0.00"}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="h-[250px]">
                        {paymentsByDayData ? (
                          <Bar
                            options={barChartOptions}
                            data={paymentsByDayData}
                          />
                        ) : (
                          <div className="flex items-center justify-center h-full">
                            <p className="text-muted-foreground">
                              No data available
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="h-[300px]">
                      {monthlyPaymentsData ? (
                        <Bar
                          options={barChartOptions}
                          data={monthlyPaymentsData}
                        />
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <p className="text-muted-foreground">
                            No data available
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
      <Dialog open={editNameModalOpen} onOpenChange={setEditNameModalOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Edit User Name</DialogTitle>
          </DialogHeader>
          <form onSubmit={handleEditNameSubmit}>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="firstName" className="text-right">
                  First Name
                </Label>
                <Input
                  id="firstName"
                  value={editNameForm.firstName}
                  onChange={(e) =>
                    setEditNameForm({
                      ...editNameForm,
                      firstName: e.target.value,
                    })
                  }
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="lastName" className="text-right">
                  Last Name
                </Label>
                <Input
                  id="lastName"
                  value={editNameForm.lastName}
                  onChange={(e) =>
                    setEditNameForm({
                      ...editNameForm,
                      lastName: e.target.value,
                    })
                  }
                  className="col-span-3"
                />
              </div>
            </div>
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setEditNameModalOpen(false)}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={editUserNamesMutation.isPending}>
                {editUserNamesMutation.isPending ? "Saving..." : "Save Changes"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default AdminDashboard;
