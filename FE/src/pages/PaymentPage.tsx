import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CreditCard, Lock, Loader2 } from "lucide-react";
import { appointmentApi, Appointment } from "@/api/appointment";
import { format } from "date-fns";
import {loadStripe} from '@stripe/stripe-js';

const PaymentPage = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  
  // Fetch appointment data using React Query
  const { data: appointment, isLoading, error } = useQuery({
    queryKey: ["appointment", id],
    queryFn: () => appointmentApi.getAppointmentById(id!),
    enabled: !!id,
  });

  // Create Stripe checkout session mutation
  const createCheckoutSession = useMutation({
    mutationFn: async (appointmentId: string) => appointmentApi.checkoutSession(appointmentId),
    onSuccess: async (data) => {
      const stripe = await loadStripe(import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY!);

      // Redirect to Stripe Checkout
      const result = stripe?.redirectToCheckout({
        sessionId: data.sessionId ?? '',
      })
    },
  });

  const handlePayment = () => {
    if (id) {
      createCheckoutSession.mutate(id);
    }
  };

  // Format date and time
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return format(date, "MMMM d, yyyy");
  };

  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr);
    return format(date, "h:mm a");
  };

  // Get doctor name
  const getDoctorName = (appointment: Appointment) => {
    const { Doctor } = appointment;
    if (Doctor.firstName && Doctor.lastName) {
      return `Dr. ${Doctor.firstName} ${Doctor.lastName}`;
    } else if (Doctor.firstName) {
      return `Dr. ${Doctor.firstName}`;
    } else if (Doctor.lastName) {
      return `Dr. ${Doctor.lastName}`;
    } else {
      return Doctor.userName;
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-20">
        <div className="flex justify-center items-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </div>
    );
  }

  // Error state
  if (error || !appointment) {
    return (
      <div className="container mx-auto px-4 py-20">
        <div className="max-w-3xl mx-auto">
          <h1 className="text-3xl font-bold mb-6">Payment</h1>
          <div className="text-center text-red-500 py-10">
            Failed to load appointment details. Please try again later.
          </div>
          <div className="text-center">
            <Button onClick={() => navigate("/appointments")}>
              Back to Appointments
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-20">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">Payment</h1>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Payment Form */}
          <div className="md:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Stripe Payment</CardTitle>
                <CardDescription>
                  You'll be redirected to Stripe's secure payment page to complete your transaction
                </CardDescription>
              </CardHeader>

              <CardContent className="space-y-4">
                <div className="bg-gray-50 p-6 rounded-lg border border-gray-100">
                  <div className="flex items-center justify-center mb-4">
                    <CreditCard className="h-12 w-12 text-primary" />
                  </div>
                  <p className="text-center text-sm text-muted-foreground mb-4">
                    Click the button below to proceed to Stripe's secure checkout page where you can safely enter your payment details.
                  </p>
                  <div className="flex items-center justify-center text-xs text-muted-foreground mb-2">
                    <Lock className="h-3 w-3 mr-1" />
                    <span>Your payment information is securely processed by Stripe</span>
                  </div>
                </div>
              </CardContent>

              <CardFooter className="flex justify-between">
                <Button
                  variant="outline"
                  onClick={() => navigate("/appointments")}
                >
                  Cancel
                </Button>
                <Button 
                  onClick={handlePayment} 
                  disabled={createCheckoutSession.isPending}
                >
                  {createCheckoutSession.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    `Pay $20.00 with Stripe`
                  )}
                </Button>
              </CardFooter>
            </Card>
          </div>

          {/* Appointment Summary */}
          <div className="md:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle>Appointment Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Doctor:</span>
                  <span className="font-medium">
                    {getDoctorName(appointment)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Specialty:</span>
                  <span>{appointment.Doctor.education || "Specialist"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Date:</span>
                  <span>{formatDate(appointment.Schedule.startTime)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Time:</span>
                  <span>{formatTime(appointment.Schedule.startTime)}</span>
                </div>
                <div className="border-t pt-4 mt-4">
                  <div className="flex justify-between font-bold">
                    <span>Total:</span>
                    <span>$20.00</span>
                  </div>
                </div>
                <div className="text-xs text-muted-foreground flex items-center justify-center mt-4">
                  <Lock className="h-3 w-3 mr-1" />
                  <span>Secure payment powered by Stripe</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PaymentPage;
